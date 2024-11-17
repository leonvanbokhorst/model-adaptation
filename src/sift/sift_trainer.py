import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import logging
import dotenv
import os
from pathlib import Path
from tqdm import tqdm
import json
import hashlib
import pickle
import datetime
import shutil
import gc
import torch.cuda
from sentence_transformers import SentenceTransformer
from functools import wraps
import signal
from .sift_metrics import MetricsComputer, AdaptiveStoppingMetrics
from .sift_visualization import SIFTVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


@dataclass
class SIFTConfig:
    """Configuration for SIFT and test-time fine-tuning"""
    lambda_param: float = 0.1
    num_candidates: int = 5
    batch_size: int = 1
    learning_rate: float = 5e-5
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def timeout(seconds):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            
            # Set the timeout handler
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)
            
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)
            return result
        return wrapper
    return decorator


class SIFTTrainer:
    def __init__(
        self,
        llm_name: str,
        embedding_model: str,
        cache_dir: str = "cache/embeddings",
        window_size: int = 5,
        min_steps: int = 3,
        lambda_param: float = 0.1,
        max_length: Optional[int] = None,
        embedding_dim: int = 1024,
        config: Optional[SIFTConfig] = None,
    ):
        """Initialize SIFT trainer with models and configuration."""
        # Initialize config first
        self.config = config or SIFTConfig(
            lambda_param=lambda_param,
            max_length=max_length if max_length is not None else 512
        )
        
        # Set device and dtype
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.dtype = torch.float32  # MPS requires float32
            torch.mps.empty_cache()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dtype = torch.float32  # Use float32 consistently
        
        # Initialize models and tokenizer with consistent dtype
        logger.info(f"Loading models from {llm_name} to {self.device} with dtype {self.dtype}")
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=self.dtype,
            device_map={"": self.device},
            use_flash_attention_2=False  # Disable flash attention to avoid dtype issues
        )
        
        # Convert model parameters to consistent dtype
        self.model = self.model.to(dtype=self.dtype)
        
        # Initialize tokenizer and embedding model
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_model = self.embedding_model.to(self.device)
        
        # Basic parameters
        self.lambda_param = self.config.lambda_param
        self.window_size = window_size
        self.min_steps = min_steps
        self.embedding_dim = embedding_dim
        
        # Initialize tracking components
        self.metrics_computer = MetricsComputer()
        self.adaptive_stopping = AdaptiveStoppingMetrics(
            alpha=0.1, 
            window_size=self.window_size
        )
        self.visualizer = SIFTVisualizer()
        
        # Initialize buffers
        self._uncertainty_buffer = []
        self._last_loss = None
        
        # Setup directories
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Initialize scheduler (linear warmup and decay)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=1000,  # Adjust based on expected steps
            pct_start=0.1
        )
        
        # Initialize step counter
        self.current_accumulation_step = 0

    def compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute embedding using dedicated embedding model."""
        try:
            # Try cache first
            cache_key = hashlib.md5(text.encode()).hexdigest()
            cache_path = self.cache_dir / f"{cache_key}.npy"
            
            if cache_path.exists():
                embedding = np.load(cache_path)
                # Ensure 2D shape
                if len(embedding.shape) == 1:
                    embedding = embedding.reshape(1, -1)
                return embedding
            
            # Compute new embedding
            embedding = self.embedding_model.encode(
                text,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # Ensure 2D shape
            embedding_np = np.array(embedding).reshape(1, -1)

            # Cache the result
            np.save(cache_path, embedding_np)
            return embedding_np

        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return None

    @timeout(10)  # 10 second timeout for kernel computation
    def compute_kernel_uncertainty(self, x_star: str, selected_points: List[str]) -> float:
        try:
            # Get embeddings
            x_star_emb = self.compute_embedding(x_star)
            if x_star_emb is None or len(selected_points) == 0:
                return float('inf')
            
            # Ensure x_star_emb is 2D
            if len(x_star_emb.shape) == 1:
                x_star_emb = x_star_emb.reshape(1, -1)
            
            # Compute embeddings with timeout protection
            selected_embs = []
            for point in selected_points:
                emb = self.compute_embedding(point)
                if emb is None:
                    return float('inf')
                # Ensure each embedding is 2D
                if len(emb.shape) == 1:
                    emb = emb.reshape(1, -1)
                selected_embs.append(emb)
            
            # Stack embeddings and ensure correct shape
            selected_embs = np.vstack([emb.reshape(1, -1) for emb in selected_embs])
            
            # Add small epsilon for numerical stability
            epsilon = 1e-8
            
            # Compute kernel values with correct shapes
            k_xx = np.dot(x_star_emb, x_star_emb.T) + epsilon
            K_X = self.compute_kernel_matrix_batch(selected_embs) + epsilon * np.eye(len(selected_points))
            k_X = np.dot(x_star_emb, selected_embs.T)
            
            # Add regularization
            K_X_reg = K_X + self.lambda_param * np.eye(len(selected_points))
            
            try:
                # Ensure shapes match for solve operation
                if k_X.shape[1] != K_X_reg.shape[0]:
                    logger.error(f"Shape mismatch: k_X: {k_X.shape}, K_X_reg: {K_X_reg.shape}")
                    return float('inf')
                    
                # Use more stable solver with condition number check
                if np.linalg.cond(K_X_reg) > 1e10:
                    logger.warning("Poorly conditioned matrix, adding more regularization")
                    K_X_reg += 0.1 * np.eye(len(selected_points))
                    
                # Reshape for solve operation
                k_X = k_X.reshape(-1, 1)
                uncertainty = float(k_xx - k_X.T @ np.linalg.solve(K_X_reg, k_X))
                
                # Validate output
                if np.isnan(uncertainty) or np.isinf(uncertainty):
                    return float('inf')
                    
                return float(np.clip(uncertainty, 0, 10.0))  # Clip to reasonable range
                
            except np.linalg.LinAlgError as e:
                logger.warning(f"Matrix inversion failed: {e}")
                return float('inf')
                
        except Exception as e:
            logger.error(f"Error in kernel uncertainty computation: {e}")
            return float('inf')

    def select_examples_sift(
        self, prompt: str, candidates: List[str], n_examples: int = 5
    ) -> List[str]:
        """Select examples using direct similarity comparison."""
        try:
            selected = []
            prompt_emb = self.compute_embedding(prompt)
            
            if prompt_emb is None:
                logger.error("Failed to compute prompt embedding")
                return selected
            
            # Track candidates processing
            from tqdm import tqdm
            
            # Compute similarities for all candidates at once
            candidate_embeddings = []
            valid_candidates = []
            
            for candidate in tqdm(candidates, desc=f"Computing embeddings", leave=False):
                emb = self.compute_embedding(candidate)
                if emb is not None:
                    candidate_embeddings.append(emb)
                    valid_candidates.append(candidate)
            
            if not valid_candidates:
                return selected
            
            # Stack all embeddings
            candidate_embeddings = np.vstack(candidate_embeddings)
            
            # Compute similarities
            similarities = np.dot(prompt_emb, candidate_embeddings.T)[0]
            
            # Get top k examples
            top_indices = np.argsort(similarities)[-n_examples:][::-1]
            selected = [valid_candidates[i] for i in top_indices]
            
            logger.info(f"Selected {len(selected)}/{n_examples} examples")
            return selected
            
        except Exception as e:
            logger.error(f"Error in example selection: {str(e)}")
            return selected

    def compute_metrics(self, outputs) -> Dict[str, float]:
        """Compute metrics from model outputs."""
        try:
            if outputs is None:
                return None

            loss = outputs.loss.item() if hasattr(outputs, "loss") else float("inf")

            metrics = {
                "loss": loss,
                "perplexity": torch.exp(torch.tensor(loss)).item(),
                "bits_per_byte": loss / np.log(2),
                "uncertainty": self.compute_uncertainty(None),
            }

            return metrics

        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            return None

    @timeout(30)
    def fine_tune_step(self, example: str) -> Optional[Dict[str, Any]]:
        try:
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            # Tokenize
            inputs = self.tokenizer(
                example,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            
            # Move to device and ensure correct dtype
            inputs = {
                "input_ids": self._ensure_tensor_dtype(inputs["input_ids"], is_index=True),
                "attention_mask": self._ensure_tensor_dtype(inputs["attention_mask"], is_index=True)
            }
            
            # Clone labels and ensure they're long type
            labels = self._ensure_tensor_dtype(inputs["input_ids"].clone(), is_index=True)

            # Forward pass
            self.model.train()
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
            )

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            return {"loss": outputs.loss.item()}

        except Exception as e:
            logger.error(f"Error in fine-tuning step: {str(e)}")
            return None

    def compute_uncertainty(self, prompt: str) -> float:
        """Compute uncertainty based on recent losses."""
        if not self._uncertainty_buffer:
            return float("inf")

        # Use standard deviation of recent losses as uncertainty measure
        if len(self._uncertainty_buffer) > 1:
            return float(np.std(self._uncertainty_buffer))
        return float(self._uncertainty_buffer[0])

    def get_loss_history(self) -> List[float]:
        """Get the history of losses."""
        return self._uncertainty_buffer.copy()

    def reset_uncertainty(self):
        """Reset uncertainty tracking."""
        self._uncertainty_buffer = []
        self._last_loss = None

    def merge_and_upload(self) -> None:
        """Merge LoRA weights and upload model to HuggingFace Hub"""
        return
        logger.info("Starting model merge process...")

        # Merge LoRA weights with base model
        merged_model = AutoPeftModelForCausalLM.from_pretrained(
            self.model_name,  # Use stored model name instead of accessing it through the model
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        merged_model = merged_model.merge_and_unload()

        logger.info("Model merged successfully. Starting upload...")

        # Save merged model and tokenizer locally first
        merged_model.save_pretrained(MERGED_NAME)
        self.tokenizer.save_pretrained(MERGED_NAME)

        # Initialize Hugging Face API and upload
        api = HfApi(token=HF_TOKEN)
        api.create_repo(repo_id=REPO_ID, exist_ok=True, private=True)

        api.upload_folder(
            folder_path=MERGED_NAME,
            repo_id=REPO_ID,
            commit_message="Upload merged SIFT model",
        )

        logger.info(f"Model successfully uploaded to {REPO_ID}")

    def clear_cache(self):
        """Clear the embedding cache."""
        try:
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared embedding cache")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def compute_kernel_matrix_batch(self, embeddings: np.ndarray) -> np.ndarray:
        try:
            return np.dot(embeddings, embeddings.T)
        except Exception as e:
            logger.error(f"Error in kernel computation: {e}")
            return np.zeros((embeddings.shape[0], embeddings.shape[0]))

    def get_training_summary(self) -> Dict[str, List[float]]:
        """Get complete training summary for visualization."""
        return {
            "loss": self._uncertainty_buffer,
            "uncertainty": [self.compute_uncertainty(None) for _ in self._uncertainty_buffer],
            "compute": list(range(len(self._uncertainty_buffer)))
        }

    def clear_memory(self):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def should_stop_adaptive(
        self, uncertainty: float, step: int, alpha: float = 0.1
    ) -> bool:
        """Enhanced adaptive stopping criterion with stability checks."""
        try:
            if step < self.min_steps:
                return False
            
            # Use adaptive window size
            adaptive_window = self.compute_adaptive_window_size(step)
            
            # Add uncertainty to buffer
            self._uncertainty_buffer.append(uncertainty)
            
            # Keep buffer size manageable
            if len(self._uncertainty_buffer) > adaptive_window:
                self._uncertainty_buffer.pop(0)
            
            # Get recent uncertainties with outlier removal
            recent_uncertainties = np.array(self._uncertainty_buffer[-adaptive_window:])
            if len(recent_uncertainties) < adaptive_window:
                return False
            
            # Remove outliers using IQR method
            q1, q3 = np.percentile(recent_uncertainties, [25, 75])
            iqr = q3 - q1
            mask = (recent_uncertainties >= q1 - 1.5 * iqr) & (recent_uncertainties <= q3 + 1.5 * iqr)
            filtered_uncertainties = recent_uncertainties[mask]
            
            if len(filtered_uncertainties) < 3:  # Require minimum number of valid points
                return False
            
            # Compute robust statistics
            avg_uncertainty = np.median(filtered_uncertainties)
            uncertainty_std = np.std(filtered_uncertainties)
            
            # Dynamic threshold based on training progress
            threshold = alpha * (1 + 1/np.sqrt(1 + step))
            
            # Check both absolute and relative stability
            is_stable = uncertainty_std < 0.1 * avg_uncertainty
            is_low = avg_uncertainty < threshold
            
            should_stop = is_stable and is_low
            
            if should_stop:
                logger.info(f"Stopping at step {step} with uncertainty {avg_uncertainty:.4f} (threshold: {threshold:.4f})")
            
            return should_stop
            
        except Exception as e:
            logger.error(f"Error in stopping criterion: {e}")
            return False

    def save_checkpoint(self, path: str):
        """Save a checkpoint of the current model state."""
        try:
            checkpoint_dir = Path(path)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'uncertainty_buffer': self._uncertainty_buffer,
                'current_accumulation_step': getattr(self, 'current_accumulation_step', 0),
            }
            
            # Add optimizer and scheduler if they exist
            if hasattr(self, 'optimizer'):
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            if hasattr(self, 'scheduler'):
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            torch.save(checkpoint, checkpoint_dir / 'checkpoint.pt')
            logger.info(f"Saved checkpoint to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def update_ema_loss(self, current_loss: float, alpha: float = 0.1) -> float:
        """Update exponential moving average of loss."""
        if not hasattr(self, '_ema_loss'):
            self._ema_loss = current_loss
        else:
            self._ema_loss = alpha * current_loss + (1 - alpha) * self._ema_loss
        return self._ema_loss

    def update_and_visualize_metrics(
        self, 
        current_loss: float, 
        uncertainty: float, 
        step: int,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update metrics and generate visualizations."""
        try:
            # Update metrics history
            self.metrics_computer.update({
                'loss': current_loss,
                'uncertainty': uncertainty
            })
            
            # Get complete metrics history
            metrics_history = self.metrics_computer.get_metrics_summary()
            
            # Generate visualization if save path is provided
            if save_path:
                self.visualizer.plot_training_summary(
                    losses=metrics_history['loss'],
                    uncertainties=metrics_history['uncertainty'],
                    save_path=save_path
                )
            
            return {
                'loss': current_loss,
                'uncertainty': uncertainty,
                'step': step
            }
            
        except Exception as e:
            logger.error(f"Error in metrics update: {e}")
            return None

    def generate_training_summary(self, save_dir: str = "training_summary"):
        """Generate comprehensive training summary and visualizations."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Get complete training history
        summary = self.get_training_summary()
        
        # Save metrics to JSON
        with open(f"{save_dir}/metrics.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Generate final visualizations
        self.visualizer.plot_adaptive_stopping(
            metrics=summary,
            alpha=self.adaptive_stopping.alpha,
            title="Final Training Summary",
            save_path=f"{save_dir}/final_stopping_analysis.png"
        )
        
        # Save stopping points
        stopping_summary = self.adaptive_stopping.get_stopping_summary()
        with open(f"{save_dir}/stopping_points.json", "w") as f:
            json.dump(stopping_summary, f, indent=2)

    def _ensure_tensor_dtype(self, tensor: torch.Tensor, is_index: bool = False) -> torch.Tensor:
        """Ensure tensor has correct dtype."""
        if is_index:
            # For indices (input_ids, attention masks), use long
            tensor = tensor.to(dtype=torch.long)
        else:
            # For other tensors, use configured dtype
            if tensor.dtype != self.dtype:
                tensor = tensor.to(dtype=self.dtype)
        return tensor.to(self.device)

    def adjust_learning_rate(self, current_loss: float, window_size: int = 5) -> None:
        if len(self._uncertainty_buffer) >= window_size:
            recent_losses = self._uncertainty_buffer[-window_size:]
            loss_trend = np.mean(np.diff(recent_losses))
            
            # Increase learning rate if loss is stagnating
            if abs(loss_trend) < 0.01:
                self.optimizer.param_groups[0]['lr'] *= 1.2
            # Decrease learning rate if loss is unstable
            elif loss_trend > 0:
                self.optimizer.param_groups[0]['lr'] *= 0.8

    def enhanced_early_stopping(self, 
        current_loss: float, 
        patience: int = 10, 
        min_delta: float = 0.01
    ) -> bool:
        if not hasattr(self, '_best_loss'):
            self._best_loss = float('inf')
            self._patience_counter = 0
            
        if current_loss < (self._best_loss - min_delta):
            self._best_loss = current_loss
            self._patience_counter = 0
            return False
            
        self._patience_counter += 1
        return self._patience_counter >= patience

    def save_checkpoint_with_metrics(self, path: str, metrics: Dict[str, float]):
        checkpoint_dir = Path(path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'uncertainty_buffer': self._uncertainty_buffer,
            'current_step': self.current_accumulation_step,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save with metrics in filename
        filename = f"checkpoint_loss_{metrics['loss']:.4f}_step_{self.current_accumulation_step}.pt"
        torch.save(checkpoint, checkpoint_dir / filename)
        
        # Keep only top N checkpoints
        self._cleanup_old_checkpoints(checkpoint_dir, keep_top_n=3)

    def compute_adaptive_window_size(self, step: int, min_window: int = 3, max_window: int = 10) -> int:
        """Compute adaptive window size based on training progress."""
        # Start small and increase window size as training progresses
        progress_factor = min(1.0, step / 1000)  # Normalize steps to [0,1]
        window_size = min_window + int((max_window - min_window) * progress_factor)
        return window_size

    def _cleanup_old_checkpoints(self, checkpoint_dir: Path, keep_top_n: int = 3):
        """Keep only the top N checkpoints based on loss."""
        try:
            # Get all checkpoint files
            checkpoint_files = list(checkpoint_dir.glob("checkpoint_loss_*.pt"))
            
            if len(checkpoint_files) <= keep_top_n:
                return
            
            # Extract loss values from filenames
            def get_loss(filepath):
                try:
                    return float(str(filepath).split("loss_")[1].split("_")[0])
                except:
                    return float('inf')
            
            # Sort by loss and keep only top N
            sorted_files = sorted(checkpoint_files, key=get_loss)
            files_to_remove = sorted_files[keep_top_n:]
            
            # Remove excess checkpoints
            for file in files_to_remove:
                file.unlink()
            
        except Exception as e:
            logger.error(f"Error cleaning up checkpoints: {e}")

    def compute_validation_metrics(self, validation_examples: List[str]) -> Dict[str, float]:
        """Compute validation metrics on a subset of examples."""
        self.model.eval()
        total_loss = 0.0
        total_perplexity = 0.0
        
        with torch.no_grad():
            for example in validation_examples[:10]:  # Limit to 10 examples for speed
                try:
                    inputs = self.tokenizer(
                        example,
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_length,
                        return_tensors="pt"
                    )
                    
                    inputs = {k: self._ensure_tensor_dtype(v, is_index=True) 
                             for k, v in inputs.items()}
                    
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    total_loss += outputs.loss.item()
                    total_perplexity += torch.exp(outputs.loss).item()
                    
                except Exception as e:
                    logger.error(f"Error in validation: {e}")
                    continue
        
        n_examples = min(len(validation_examples), 10)
        metrics = {
            'val_loss': total_loss / max(n_examples, 1),
            'val_perplexity': total_perplexity / max(n_examples, 1)
        }
        
        self.model.train()
        return metrics


def main():
    """Test the SIFT trainer"""
    trainer = SIFTTrainer(
        llm_name="unsloth/Llama-3.2-1B",
        embedding_model="BAAI/bge-large-en-v1.5",
        cache_dir="cache/embeddings",
    )

    # Example usage
    prompt = "What is machine learning?"
    candidates = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks.",
        "Python is a programming language.",
        "Machine learning algorithms learn from data.",
        "Data science involves statistics and programming.",
    ]

    logger.info("Testing example selection...")
    selected = trainer.select_examples_sift(prompt, candidates)
    logger.info("Selected examples:")
    for i, example in enumerate(selected, 1):
        logger.info(f"{i}. {example}")

    # Save the index
    trainer.save_index()


if __name__ == "__main__":
    main()
