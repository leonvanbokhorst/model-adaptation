import os
import json
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datasets import load_dataset
from transformers import AutoTokenizer
import logging
from tqdm import tqdm

try:
    import zstandard
except ImportError:
    raise ImportError(
        "The Pile dataset requires the 'zstandard' package. "
        "Please install it with: pip install zstandard"
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataLoader:
    """Data loader for the Pile dataset with support for subsetting and batching."""

    def __init__(
        self,
        tokenizer_name: str,
        max_length: int,
        batch_size: int,
        dataset_name: str = "openwebtext",
        split: str = "train",
        streaming: bool = True,
        **dataset_args
    ):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.max_length = max_length
            self.batch_size = batch_size
            
            default_args = {"streaming": streaming}
            dataset_args = {**default_args, **dataset_args}
            
            self.dataset = load_dataset(dataset_name, split=split, **dataset_args)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize TextDataLoader: {str(e)}") from e

    def get_batches(self) -> List[Dict[str, torch.Tensor]]:
        """Get batches from the dataset."""
        batches = []
        current_batch = []

        for item in self.dataset:
            encoded = self.tokenizer(
                item["text"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            current_batch.append(encoded)

            if len(current_batch) == self.batch_size:
                batches.append(self._collate_batch(current_batch))
                current_batch = []

        if current_batch:
            batches.append(self._collate_batch(current_batch))

        return batches

    def _collate_batch(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Collate a batch of encoded inputs."""
        keys = batch[0].keys()
        collated = {key: torch.cat([b[key] for b in batch], dim=0) for key in keys}
        return collated

    def get_embeddings(self, embedding_model) -> np.ndarray:
        """Get embeddings for all texts using the provided model."""
        embeddings = []

        for batch in self.get_batches():
            with torch.no_grad():
                batch_embeddings = embedding_model(**batch).last_hidden_state[:, 0, :]
                embeddings.append(batch_embeddings.cpu().numpy())

        return np.concatenate(embeddings, axis=0)


class SubsetSampler:
    """Helper class to sample subsets from the Pile for evaluation."""

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.cached_prompts = []
        self.cached_training = []
        logger.info("SubsetSampler initialized")
        
        # Log first example format for debugging
        for example in self.dataloader.dataset:
            logger.info(f"Example format: {type(example)}")
            logger.info(f"Example content: {example}")
            break

    def _extract_text(self, example: Dict[str, Any]) -> str:
        """Extract text from dataset example."""
        if isinstance(example, dict) and 'text' in example:
            return example['text']
        return str(example)

    def sample_test_prompts(self, n_prompts=100) -> List[str]:
        """Sample prompts from streaming dataset."""
        logger.info(f"Sampling {n_prompts} test prompts...")
        
        if not self.cached_prompts:
            target_cache_size = n_prompts * 2
            logger.info(f"Building prompt cache (target size: {target_cache_size})...")
            
            for i, example in tqdm(
                enumerate(self.dataloader.dataset), 
                desc="Caching prompts",
                total=target_cache_size
            ):
                if i >= target_cache_size:
                    break
                text = self._extract_text(example)
                self.cached_prompts.append(text)
            
            logger.info(f"Cached {len(self.cached_prompts)} prompts")
        
        # Randomly sample from cached examples
        indices = np.random.choice(len(self.cached_prompts), n_prompts, replace=False)
        sampled_prompts = [self.cached_prompts[i] for i in indices]
        logger.info(f"Selected {len(sampled_prompts)} test prompts")
        
        return sampled_prompts

    def get_training_subset(self, size=1000) -> List[str]:
        """Get training examples from streaming dataset."""
        logger.info(f"Getting training subset of size {size}...")
        
        if not self.cached_training:
            target_cache_size = size * 2
            logger.info(f"Building training cache (target size: {target_cache_size})...")
            
            for i, example in tqdm(
                enumerate(self.dataloader.dataset), 
                desc="Caching training examples",
                total=target_cache_size
            ):
                if i >= target_cache_size:
                    break
                text = self._extract_text(example)
                self.cached_training.append(text)
            
            logger.info(f"Cached {len(self.cached_training)} training examples")
        
        # Randomly sample from cached examples
        indices = np.random.choice(len(self.cached_training), size, replace=False)
        sampled_training = [self.cached_training[i] for i in indices]
        logger.info(f"Selected {len(sampled_training)} training examples")
        
        # Log a sample to verify data format
        if sampled_training:
            logger.debug(f"Sample training example: {sampled_training[0][:100]}...")
        
        return sampled_training

if __name__ == "__main__":
    # Test/example code
    print("Testing PileDataLoader and SubsetSampler...")
    
    # Initialize the data loader
    data_loader = TextDataLoader(
        tokenizer_name="unsloth/Llama-3.2-1B",
        max_length=512,
        batch_size=32,
        dataset_name="openwebtext",
        split="train",
        streaming=True
    )
    
    # Initialize sampler
    sampler = SubsetSampler(data_loader)
    
    # Test sampling
    print("\nSampling test prompts...")
    test_prompts = sampler.sample_test_prompts(n_prompts=2)
    print(f"Number of test prompts: {len(test_prompts)}")
    print("First prompt sample:", test_prompts[0])
    
    print("\nGetting training subset...")
    training_data = sampler.get_training_subset(size=2)
    print(f"Number of training examples: {len(training_data)}")
    print("First training sample:", training_data[0])
