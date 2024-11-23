"""
Understanding Attention Mechanisms in Neural Networks

Attention mechanisms are a fundamental concept in modern deep learning, especially in 
transformers. This example demonstrates a simple attention mechanism that can:
1. Encode sentences into vector representations
2. Calculate attention scores between words
3. Find relevant sentences based on attention

Historical Significance:
- Introduced in 2014 by Bahdanau et al. for machine translation
- Revolutionary because it allowed models to "focus" on relevant parts of input
- Led directly to the transformer architecture (2017) and modern LLMs

Key Concepts Demonstrated:
1. Word Embeddings: Converting words to vectors
2. Attention Scores: Measuring relevance between vectors
3. Dot Product Attention: Simplest form of attention mechanism

The network below uses:
- Word embeddings (5 dimensions per word)
- Simple dot product attention
- Mean pooling for sentence-level scores
"""

import torch
import torch.nn as nn


class CoolAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # Our dataset: simple sentences about food preferences
        # Each sentence follows pattern: [Person] [Verb] [Food]
        self.story = [
            "Alice loves pizza",
            "Bob hates broccoli",
            "Charlie eats cookies",
            "Alice likes cake",
            "Bob loves sushi",
        ]

        # Vocabulary mapping: convert words to unique indices
        # Organized by semantic categories (people, verbs, foods)
        self.word2idx = {
            # People embeddings (indices 0-2)
            "Alice": 0,
            "Bob": 1,
            "Charlie": 2,
            # Verb embeddings (indices 3-6)
            "loves": 3,
            "hates": 4,
            "likes": 5,
            "eats": 6,
            # Food embeddings (indices 7-11)
            "pizza": 7,
            "broccoli": 8,
            "cookies": 9,
            "cake": 10,
            "sushi": 11,
        }

        # Create learnable word embeddings
        # - Each word gets a 5-dimensional vector
        # - These vectors are randomly initialized and could be trained
        # - 5 dimensions is arbitrary (could be larger for more complex relationships)
        self.embeddings = nn.Embedding(len(self.word2idx), 5)

    def encode_sentence(self, sentence):
        """
        Convert a sentence into its vector representation.
        
        Args:
            sentence (str): Input sentence to encode
            
        Returns:
            torch.Tensor: Tensor of word embeddings (shape: [num_words, embedding_dim])
        """
        # Split sentence into words and convert to indices
        words = sentence.split()
        indices = [self.word2idx[word] for word in words]
        # Look up embeddings for each word
        return self.embeddings(torch.tensor(indices))

    def attention_search(self, person):
        """
        Find sentences relevant to a specific person using attention.
        
        Args:
            person (str): Person to search for
            
        Returns:
            list: Sorted list of (sentence, attention_score) tuples
        """
        results = []

        # Step 1: Convert all sentences to vector representations
        encoded_sentences = [self.encode_sentence(s) for s in self.story]

        # Step 2: Calculate attention scores for each sentence
        for i, sentence_embedding in enumerate(encoded_sentences):
            # Calculate attention using dot product between:
            # - First word of sentence (usually the person)
            # - Embedding of the search query (person)
            score = torch.mean(
                sentence_embedding[0]
                * self.embeddings(torch.tensor([self.word2idx[person]]))
            )
            results.append((self.story[i], score.item()))

        # Step 3: Sort results by attention score (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# Demo the attention mechanism
attention = CoolAttention()

# Search for Alice's food preferences
print("🔍 Searching for Alice's food preferences...")
results = attention.attention_search("Alice")
for sentence, score in results:
    # Visualize attention scores with stars
    attention_emojis = "🌟" * int(score * 5)
    print(f"{attention_emojis} {sentence}")

# Search for Bob's food preferences
print("\n🔍 Now searching for Bob's food preferences...")
results = attention.attention_search("Bob")
for sentence, score in results:
    attention_emojis = "🌟" * int(score * 5)
    print(f"{attention_emojis} {sentence}")
