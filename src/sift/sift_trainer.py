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

logging.basicConfig(level=logging.ERROR)
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
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


class SIFTTrainer:
    def __init__(
        self,
        llm_name: str = "unsloth/Llama-3.2-1B",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        index_dir: str = "cache/faiss",
        max_length: int = 512,
        embedding_dim: int = 1024,
        uncertainty_buffer_size: int = 100,
        gradient_accumulation_steps: int = 8,
        max_grad_norm: float = 1.0,
    ):
        """Initialize trainer with all required attributes."""
        self.device = torch.device("cpu")
        self.embedding_dim = embedding_dim  # Store embedding dimension
        self.max_length = max_length

        # Initialize paths
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "embeddings.faiss"
        self.metadata_path = self.index_dir / "metadata.pkl"

        # Initialize FAISS index and mappings
        self.index, self.text_to_id, self.id_to_text = self._load_or_create_index()

        # Load embedding model
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )

        self.model.eval()

        # Training settings
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.current_accumulation_step = 0

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=5e-5, weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        self.max_grad_norm = max_grad_norm

        # Initialize uncertainty tracking
        self._uncertainty_buffer = []
        self._uncertainty_buffer_size = uncertainty_buffer_size
        self._last_loss = None

        logger.info(f"Using device: {self.device}")
        logger.info(f"Model loaded with dtype: {self.model.dtype}")

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
            # Check if text is already in index
            if text in self.text_to_id:
                vector_id = self.text_to_id[text]
                reconstructed = np.zeros((1, self.embedding_dim), dtype=np.float32)
                self.index.reconstruct(vector_id, reconstructed[0])
                return reconstructed

            # Compute new embedding
            embedding = self.embedding_model.encode(
                text,
                normalize_embeddings=True,  # Important for cosine similarity
                show_progress_bar=False,
            )

            # Reshape for FAISS
            embedding_np = np.array(embedding).reshape(1, -1)

            # Add to index
            vector_id = self.index.ntotal
            self.index.add(embedding_np)
            self.text_to_id[text] = vector_id
            self.id_to_text[vector_id] = text

            return embedding_np

        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return None

    def select_examples(
        self, prompt: str, candidates: List[str], n_examples: int = 5
    ) -> List[str]:
        """Select examples using FAISS index."""
        try:
            # Compute prompt embedding
            prompt_embedding = self.compute_embedding(prompt)
            if prompt_embedding is None:
                raise ValueError("Failed to compute prompt embedding")

            # Add candidates to index if not already present
            for candidate in tqdm(candidates, desc="Processing candidates"):
                if candidate not in self.text_to_id:
                    self.compute_embedding(candidate)

            # Search for nearest neighbors
            D, I = self.index.search(prompt_embedding, n_examples)

            # Get corresponding texts
            selected = [self.id_to_text[int(i)] for i in I[0]]
            logger.info(f"Selected {len(selected)} examples using FAISS")

            # Save updated index
            self.save_index()

            return selected

        except Exception as e:
            logger.error(f"Error in example selection: {str(e)}")
            logger.error("Falling back to random selection")
            indices = np.random.choice(len(candidates), n_examples, replace=False)
            return [candidates[idx] for idx in indices]

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

    def fine_tune_step(self, example: str) -> Optional[Dict[str, Any]]:
        """Perform fine-tuning step with proper metrics."""
        try:
            gc.collect()

            # Tokenize
            inputs = self.tokenizer(
                example,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            labels = inputs["input_ids"].clone()

            # Forward pass
            self.model.train()
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels,
            )

            # Compute loss and backward
            loss = outputs.loss / self.gradient_accumulation_steps
            loss.backward()

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Update weights if needed
            self.current_accumulation_step += 1
            if self.current_accumulation_step >= self.gradient_accumulation_steps:
                self.optimizer.step()

                # Update learning rate based on loss
                self.scheduler.step(loss.item())

                self.optimizer.zero_grad()
                self.current_accumulation_step = 0

            self.model.eval()

            # Update uncertainty
            loss_value = loss.item() * self.gradient_accumulation_steps
            self._last_loss = loss_value
            self._uncertainty_buffer.append(loss_value)
            if len(self._uncertainty_buffer) > self._uncertainty_buffer_size:
                self._uncertainty_buffer.pop(0)

            # Compute metrics
            metrics = self.compute_metrics(outputs)
            if metrics is None:
                return None

            # Add training info
            metrics.update(
                {
                    "labels": labels.detach(),
                    "logits": (
                        outputs.logits.detach() if hasattr(outputs, "logits") else None
                    ),
                }
            )

            # Clear memory
            del outputs
            gc.collect()

            return metrics

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


class MetricsComputer:
    def __init__(self):
        self.metrics_history = {
            "bits_per_byte": [],
            "perplexity": [],
            "uncertainty": [],
        }

    def compute_metrics(
        self, outputs: Dict[str, Any], uncertainty: float = None
    ) -> Dict[str, float]:
        """Compute and store metrics."""
        if outputs is None:
            return None

        metrics = {
            "bits_per_byte": outputs.get("bits_per_byte", 0.0),
            "perplexity": outputs.get("perplexity", 0.0),
            "uncertainty": outputs.get(
                "uncertainty", uncertainty if uncertainty is not None else 0.0
            ),
        }

        # Store metrics
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        return metrics

    def get_metrics_summary(self) -> Dict[str, List[float]]:
        """Get the history of all metrics."""
        return self.metrics_history


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
    selected = trainer.select_examples(prompt, candidates)
    logger.info("Selected examples:")
    for i, example in enumerate(selected, 1):
        logger.info(f"{i}. {example}")

    # Save the index
    trainer.save_index()


if __name__ == "__main__":
    main()
