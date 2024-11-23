"""
Word2Vec: Understanding Words Through Context

Word2Vec, introduced by Mikolov et al. at Google in 2013, revolutionized how computers understand
words by learning their meaning from context. The key insight was: words that appear in similar
contexts probably have similar meanings.

For example, in these sentences:
- "The cat drinks milk"
- "The dog drinks water"
We can guess that 'cat' and 'dog' are similar because they appear in similar contexts.

The model works by:
1. Converting each word to a dense vector (embedding)
2. Learning to predict context words from target words (or vice versa)
3. Similar words end up with similar vectors

Two main architectures:
- Skip-gram: Predict context words from target word
- CBOW (Continuous Bag of Words): Predict target word from context words

This implementation uses Skip-gram with negative sampling:
- For each word, look at nearby words (within a window)
- Learn to predict these context words (positive samples)
- Also learn to NOT predict random other words (negative samples)

The resulting word embeddings capture semantic relationships:
king - man + woman ≈ queen
paris - france + italy ≈ rome
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter, deque
import random


class Word2Vec(nn.Module):
    """
    Neural network for learning word embeddings.
    Uses two embedding layers:
    - target_embeddings: for the main word we're looking at
    - context_embeddings: for the surrounding words
    """

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # Two separate embedding layers:
        # - When a word is the target, we use target_embeddings
        # - When a word is in the context, we use context_embeddings
        # This asymmetry helps learn richer representations
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Initialize with small random values to break symmetry
        # Without this, all words would start too similar
        self.target_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.context_embeddings.weight.data.uniform_(-0.1, 0.1)

    def forward(self, target_word, context_word):
        # Get vector representations
        target_embed = self.target_embeddings(target_word)
        context_embed = self.context_embeddings(context_word)

        # Compute similarity using dot product
        # Similar words should have vectors pointing in similar directions
        similarity = torch.sum(target_embed * context_embed, dim=1)

        return torch.sigmoid(similarity)

    def get_embedding(self, word_idx):
        # For using the trained model, we only need target embeddings
        # Context embeddings are just for training
        return self.target_embeddings(torch.tensor([word_idx])).detach()


class Word2VecTrainer:
    """
    Handles the training process for Word2Vec:
    1. Creates vocabulary from text
    2. Generates training pairs (target word + context)
    3. Trains the model using negative sampling
    """

    def __init__(self, text, embedding_dim=64, window_size=2, min_count=5):
        self.window_size = window_size  # How many words to look at on each side

        # Create vocabulary from text
        words = text.lower().split()
        word_counts = Counter(words)

        # Filter out rare words (appear less than min_count times)
        # This reduces noise and speeds up training
        filtered_words = [
            (word, count) for word, count in word_counts.items() if count >= min_count
        ]

        # Create word-to-index mappings
        self.vocab = {word: idx for idx, (word, _) in enumerate(filtered_words)}
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # Generate training pairs
        self.training_pairs = self._create_training_pairs(words)

        # Initialize model and training tools
        self.model = Word2Vec(self.vocab_size, embedding_dim)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.BCELoss()

    def _create_training_pairs(self, words):
        """
        Creates training pairs using sliding window approach:
        - For each word (target), look at nearby words (context)
        - Create positive pairs (target + actual context word)
        - Create negative pairs (target + random word)
        """
        pairs = []
        window = deque(maxlen=2 * self.window_size + 1)

        for word in words:
            if word in self.vocab:
                window.append(word)
                if len(window) == 2 * self.window_size + 1:
                    target = window[self.window_size]  # Middle word
                    # Get context words (words before and after target)
                    context = (
                        list(window)[: self.window_size]
                        + list(window)[self.window_size + 1 :]
                    )

                    for ctx_word in context:
                        if ctx_word in self.vocab:
                            # Positive pair: target word + context word (label = 1)
                            pairs.append(
                                (self.vocab[target], self.vocab[ctx_word], 1.0)
                            )

                            # Negative pair: target word + random word (label = 0)
                            # Keep sampling until we get a word not in current context
                            neg_idx = random.randint(0, self.vocab_size - 1)
                            while self.idx_to_word[neg_idx] in context + [target]:
                                neg_idx = random.randint(0, self.vocab_size - 1)

                            pairs.append((self.vocab[target], neg_idx, 0.0))
        return pairs

    def train(self, epochs=100, batch_size=24):
        """
        Trains the model using mini-batch gradient descent:
        1. Split data into batches
        2. For each batch:
           - Make predictions
           - Calculate loss
           - Update model weights
        """
        print(f"Training Word2Vec model with {self.vocab_size} words...")
        for epoch in range(epochs):
            total_loss = 0
            # Shuffle pairs to prevent learning order dependencies
            random.shuffle(self.training_pairs)

            # Process in batches for efficiency
            for i in range(0, len(self.training_pairs), batch_size):
                batch = self.training_pairs[i : i + batch_size]
                targets, contexts, labels = zip(*batch)

                # Convert to PyTorch tensors
                target_tensor = torch.tensor(targets)
                context_tensor = torch.tensor(contexts)
                label_tensor = torch.tensor(labels, dtype=torch.float32)

                # Training step
                self.optimizer.zero_grad()  # Reset gradients
                outputs = self.model(target_tensor, context_tensor)  # Forward pass
                loss = self.criterion(outputs, label_tensor)  # Calculate loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights

                total_loss += loss.item()

            # Print progress
            avg_loss = total_loss / (len(self.training_pairs) / batch_size)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def get_similar_words(self, word, n=5):
        """
        Finds words with similar meanings by:
        1. Getting the target word's embedding
        2. Computing similarity with all other words
        3. Returning the most similar ones
        """
        if word not in self.vocab:
            return []

        # Get embedding for input word
        word_embedding = self.model.get_embedding(self.vocab[word])

        # Compare with all other words using cosine similarity
        similarities = []
        for other_word, idx in self.vocab.items():
            if other_word != word:
                other_embedding = self.model.get_embedding(idx)
                similarity = torch.cosine_similarity(word_embedding, other_embedding)
                similarities.append((other_word, similarity.item()))

        # Return top N most similar words
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]


if __name__ == "__main__":
    # Example text with related concepts
    text = """
    The quick brown fox jumps over the lazy dog.
    The fox is quick and brown and jumps high.
    The dog is lazy and sleeps all day.
    The quick rabbit jumps over the fence.
    The brown bear likes honey and fish.
    Fish swim in the river all day.
    Dogs and foxes are related animals.
    Bears and foxes live in the forest.
    """

    # Create and train model
    trainer = Word2VecTrainer(
        text,
        embedding_dim=64,  # Size of word vectors
        window_size=2,  # Words to consider as context
        min_count=2,  # Minimum word frequency
    )

    trainer.train(epochs=100, batch_size=24)

    # Test the model by finding similar words
    test_words = ["quick", "fox", "dog", "river", "bear"]
    for word in test_words:
        similar = trainer.get_similar_words(word)
        print(f"\nWords similar to '{word}':")
        for similar_word, similarity in similar:
            if similarity > 0.4:
                print(f"  {similar_word}: {similarity:.3f}")
