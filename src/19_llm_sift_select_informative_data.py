import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
import dotenv
import os
from huggingface_hub import HfApi
from peft import AutoPeftModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_NAME = "unsloth/Llama-3.2-1B"
MERGED_NAME = "Llama-3.2-1B-SIFT"
REPO_ID = f"leonvanbokhorst/{MERGED_NAME}"


@dataclass
class SIFTConfig:
    """Configuration for SIFT and test-time fine-tuning"""

    lambda_param: float = 0.1  # Trade-off parameter for SIFT
    num_candidates: int = 5    # Reduced from 200 to a more reasonable default
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
        model_name: str = MODEL_NAME,
        config: Optional[SIFTConfig] = None,
    ):
        self.config = config or SIFTConfig()
        self.model_name = model_name  # Store the model name for later use

        # Simplified model loading for MPS
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

        # Skip PEFT preparation on MPS
        if self.config.device == "mps":
            logger.info("Skipping PEFT preparation on MPS device")
        else:
            self.model = prepare_model_for_kbit_training(self.model)

        # LoRA configuration
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info(f"Model loaded on {self.config.device}")

    def compute_embeddings(self, texts: List[str]) -> torch.Tensor:
        """Compute embeddings for a list of texts"""
        embeddings = []

        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(
                    text,
                    max_length=self.config.max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(self.config.device)

                # Get the last hidden state
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use the last layer's [CLS] token embedding
                embedding = outputs.hidden_states[-1][:, 0, :].cpu()
                embeddings.append(embedding)

        return torch.cat(embeddings, dim=0)

    def compute_kernel_matrix(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute the kernel matrix using cosine similarity"""
        n = embeddings.shape[0]
        K = torch.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                K[i, j] = F.cosine_similarity(
                    embeddings[i:i+1], 
                    embeddings[j:j+1], 
                    dim=1
                )
        
        return K

    def select_examples(
        self, prompt: str, candidates: List[str], num_select: Optional[int] = None
    ) -> List[int]:
        """Select examples using SIFT algorithm"""
        if not candidates:
            raise ValueError("No candidates provided")
        
        num_select = min(
            num_select or self.config.num_candidates,
            len(candidates)
        )
        
        logger.info(f"Selecting {num_select} examples from {len(candidates)} candidates")
        
        prompt_embedding = self.compute_embeddings([prompt])
        candidate_embeddings = self.compute_embeddings(candidates)

        selected_indices = []

        for i in range(num_select):
            logger.info(f"Selection iteration {i+1}/{num_select}")
            
            if len(selected_indices) == 0:  # Changed condition to check length
                # First selection: use cosine similarity
                similarities = F.cosine_similarity(
                    prompt_embedding, candidate_embeddings, dim=1
                )
                next_idx = similarities.argmax().item()
                logger.info(f"First selection: chose index {next_idx}")
                selected_indices.append(next_idx)  # Add to selected indices immediately
                continue  # Skip to next iteration
            
            # SIFT selection for subsequent examples
            selected_embeddings = candidate_embeddings[selected_indices]
            
            # Include prompt embedding in the kernel matrix calculation
            all_embeddings = torch.cat([selected_embeddings, prompt_embedding])
            K = self.compute_kernel_matrix(all_embeddings)

            min_uncertainty = float("inf")
            next_idx = -1

            remaining_indices = [i for i in range(len(candidates)) if i not in selected_indices]
            logger.info(f"Remaining indices: {remaining_indices}")
            
            if not remaining_indices:
                break

            for idx in remaining_indices:
                candidate_embedding = candidate_embeddings[idx:idx + 1]
                
                # Compute similarity vector
                k_x = torch.zeros(K.shape[0], 1)
                for j in range(K.shape[0]):
                    k_x[j] = F.cosine_similarity(
                        candidate_embedding,
                        all_embeddings[j:j+1],
                        dim=1
                    )

                try:
                    # Solve the linear system
                    A = K + self.config.lambda_param * torch.eye(K.shape[0])
                    b = k_x
                    
                    solution = torch.linalg.solve(A, b)
                    uncertainty = 1 - (k_x.t() @ solution).item()

                    if uncertainty < min_uncertainty:
                        min_uncertainty = uncertainty
                        next_idx = idx
                        logger.info(f"New best uncertainty {uncertainty:.4f} at index {idx}")

                except Exception as e:
                    logger.warning(f"Error computing uncertainty for index {idx}: {str(e)}")
                    continue

            if next_idx != -1:
                logger.info(f"Adding index {next_idx} to selected indices")
                selected_indices.append(next_idx)
            else:
                logger.warning("No valid next index found")
                break

        logger.info(f"Final selected indices: {selected_indices}")
        return selected_indices

    def fine_tune_step(self, text: str) -> float:
        """Perform a single step of test-time fine-tuning"""
        inputs = self.tokenizer(
            text,
            max_length=self.config.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.config.device)

        outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        # Gradient step
        loss.backward()

        # Update parameters
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data -= self.config.learning_rate * param.grad
                    param.grad = None

        return loss.item()

    def merge_and_upload(self) -> None:
        """Merge LoRA weights and upload model to HuggingFace Hub"""
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
            commit_message="Upload merged SIFT model"
        )
        
        logger.info(f"Model successfully uploaded to {REPO_ID}")


def main():
    # Initialize trainer
    trainer = SIFTTrainer()

    # Example prompt and candidates
    prompt = "What is the capital of France?"
    candidates = [
        "Paris is the capital city of France.",
        "The Eiffel Tower is located in Paris.",
        "France is a country in Europe.",
        "Paris has been France's capital since 508 CE.",
        "The city of Paris hosts many government buildings.",
    ]

    logger.info(f"Starting selection with {len(candidates)} candidates")
    
    # Select examples using SIFT, limiting to available candidates
    selected_indices = trainer.select_examples(
        prompt, 
        candidates,
        num_select=3  # Explicitly select 3 examples
    )

    # Fine-tune on selected examples
    for idx in selected_indices:
        logger.info(f"Fine-tuning on example {idx}: {candidates[idx]}")
        loss = trainer.fine_tune_step(candidates[idx])
        logger.info(f"Fine-tuning loss for example {idx}: {loss:.4f}")

    # After fine-tuning, merge and upload the model
    # trainer.merge_and_upload()


if __name__ == "__main__":
    main()
