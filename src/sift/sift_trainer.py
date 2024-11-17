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
import faiss
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
        index_dir: str = "cache/faiss",
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
        
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "embeddings.faiss"
        self.metadata_path = self.index_dir / "metadata.pkl"
        
        # Load or create FAISS index
        self.index, self.text_to_id, self.id_to_text = self._load_or_create_index()
        
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

    def _load_or_create_index(
        self,
    ) -> Tuple[faiss.Index, Dict[str, int], Dict[int, str]]:
        """Load existing FAISS index or create new one."""
        if self.index_path.exists() and self.metadata_path.exists():
            try:
                # Load FAISS index
                index = faiss.read_index(str(self.index_path))

                # Load metadata
                with open(self.metadata_path, "rb") as f:
                    metadata = pickle.load(f)
                    text_to_id = metadata["text_to_id"]
                    id_to_text = metadata["id_to_text"]

                logger.info(f"Loaded existing index with {index.ntotal} vectors")
                return index, text_to_id, id_to_text
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}. Creating new one.")

        # Create new index and mappings
        index = faiss.IndexFlatL2(self.embedding_dim)
        text_to_id = {}
        id_to_text = {}
        logger.info("Created new FAISS index")
        return index, text_to_id, id_to_text

    def save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))

            # Save metadata
            metadata = {"text_to_id": self.text_to_id, "id_to_text": self.id_to_text}
            with open(self.metadata_path, "wb") as f:
                pickle.dump(metadata, f)

            logger.info(f"Saved index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

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
        """Select examples using SIFT with improved robustness and logging."""
        try:
            selected = []
            prompt_emb = self.compute_embedding(prompt)
            
            if prompt_emb is None:
                logger.error("Failed to compute prompt embedding")
                return selected
            
            # Add progress tracking
            from tqdm import tqdm
            
            for i in range(n_examples):
                min_uncertainty = float("inf")
                best_candidate = None
                
                # Track candidates processing
                for candidate in tqdm(candidates, desc=f"Processing candidates for example {i+1}/{n_examples}", leave=False):
                    if candidate in selected:
                        continue
                        
                    try:
                        # Compute uncertainty with timeout protection
                        test_selected = selected + [candidate]
                        uncertainty = self.compute_kernel_uncertainty(prompt, test_selected)
                        
                        # Skip invalid uncertainties
                        if uncertainty is None or np.isnan(uncertainty) or np.isinf(uncertainty):
                            continue
                            
                        if uncertainty < min_uncertainty:
                            min_uncertainty = uncertainty
                            best_candidate = candidate
                            
                    except Exception as e:
                        logger.warning(f"Error processing candidate: {str(e)}")
                        continue
                
                # Check if we found a valid candidate
                if best_candidate:
                    selected.append(best_candidate)
                    logger.info(f"Selected example {i+1}/{n_examples} with uncertainty: {min_uncertainty:.4f}")
                else:
                    logger.warning(f"No valid candidate found for example {i+1}")
                    break
                    
                # Clear memory periodically
                if i % 2 == 0:
                    self.clear_memory()
            
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
            
            # Add uncertainty to buffer
            self._uncertainty_buffer.append(uncertainty)
            
            # Keep buffer size manageable
            if len(self._uncertainty_buffer) > self.window_size:
                self._uncertainty_buffer.pop(0)
            
            # Get recent uncertainties with outlier removal
            recent_uncertainties = np.array(self._uncertainty_buffer[-self.window_size:])
            if len(recent_uncertainties) < self.window_size:
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
        # Update EMA loss
        ema_loss = self.update_ema_loss(current_loss)
        
        # Compute metrics
        metrics = self.metrics_computer.compute_metrics(
            {'loss': current_loss}, 
            uncertainty=uncertainty
        )
        
        # Check stopping condition using both metrics
        should_stop = self.adaptive_stopping.should_stop(
            uncertainty=uncertainty,
            step=step,
            loss_history=self.metrics_computer.metrics_history['loss']
        )
        
        # Generate visualizations periodically
        if step % 10 == 0:
            self.visualizer.plot_adaptive_stopping(
                metrics=self.get_training_summary(),
                alpha=self.adaptive_stopping.alpha,
                title=f"Training Progress - Step {step}",
                save_path=save_path
            )
        
        return {
            'metrics': metrics,
            'should_stop': should_stop,
            'ema_loss': ema_loss
        }

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


def main():
    """Test the SIFT trainer"""
    trainer = SIFTTrainer(
        llm_name="unsloth/Llama-3.2-1B",
        embedding_model="BAAI/bge-large-en-v1.5",
        index_dir="cache/faiss",
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
