import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from collections import Counter, defaultdict
import re


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_size = config.head_size
        self.dropout = config.dropout

        # Create the query, key, and value projections for all heads
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        # Output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # Causal mask to ensure we only attend to previous tokens
        self.register_buffer(
            "mask", torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Split into heads and move head dimension to be next to batch
        q = self.query(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_size).transpose(1, 2)

        # Compute attention scores ("affinities")
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, p=self.dropout, training=self.training)

        # Apply attention to values
        out = att @ v

        # Restore time as dimension 1
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            # First we expand
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            # Then we shrink back down
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # Attention with residual connection
        x = x + self.attention(self.ln1(x))
        # Feed forward with residual connection
        x = x + self.feed_forward(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Store config as instance variable
        self.config = config

        # Token embedding table
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # Position embedding table
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Get token embeddings
        tok_emb = self.token_embedding(idx)
        # Get position embeddings
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        # Combine them
        x = tok_emb + pos_emb

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Apply final layer norm
        x = self.ln_f(x)

        # Get logits
        logits = self.lm_head(x)

        # If we have targets, compute the loss
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, sample_fn=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop context to block_size
            context = idx[:, -self.config.block_size:]
            # Get predictions
            logits = self(context)
            # Focus only on the last time step
            logits = logits[:, -1, :]
            
            # Use custom sampling function if provided, otherwise default sampling
            if sample_fn is not None:
                idx_next = sample_fn(logits)
            else:
                # Default sampling logic
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# Configuration class to hold hyperparameters
class GPTConfig:
    def __init__(
        self,
        vocab_size,
        block_size,
        n_layer=6,
        n_embd=384,
        num_heads=6,
        head_size=64,
        dropout=0.1,
    ):
        self.vocab_size = vocab_size  # Size of vocabulary
        self.block_size = block_size  # Maximum sequence length
        self.n_layer = n_layer  # Number of transformer blocks
        self.n_embd = n_embd  # Embedding dimension
        self.num_heads = num_heads  # Number of attention heads
        self.head_size = head_size  # Size of each attention head
        self.dropout = dropout  # Dropout probability


import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm


class CharacterTokenizer:
    def __init__(self):
        # Special tokens with meaningful names
        self.special_tokens = {
            'PAD': '<|pad|>',
            'BOS': '<|bos|>',  # Beginning of sequence
            'EOS': '<|eos|>',  # End of sequence
            'UNK': '<|unk|>'   # Unknown token
        }
        
        # Initialize vocabularies
        self.char_to_idx = {token: idx for idx, token in enumerate(self.special_tokens.values())}
        self.idx_to_char = {idx: token for idx, token in enumerate(self.special_tokens.values())}
        self.vocab_size = len(self.special_tokens)
        
        # Store special token indices for easy access
        self.pad_idx = self.char_to_idx[self.special_tokens['PAD']]
        self.bos_idx = self.char_to_idx[self.special_tokens['BOS']]
        self.eos_idx = self.char_to_idx[self.special_tokens['EOS']]
        self.unk_idx = self.char_to_idx[self.special_tokens['UNK']]

    def fit(self, text):
        """Build vocabulary from text."""
        # Find all unique characters
        unique_chars = sorted(set(text))
        
        # Add to vocabulary if not already present
        for char in unique_chars:
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
        
        self.vocab_size = len(self.char_to_idx)
        return self

    def encode(self, text, add_special_tokens=True):
        """Convert text to token indices."""
        indices = []
        
        if add_special_tokens:
            indices.append(self.bos_idx)
            
        for char in text:
            # Use UNK token for unknown characters
            idx = self.char_to_idx.get(char, self.unk_idx)
            indices.append(idx)
            
        if add_special_tokens:
            indices.append(self.eos_idx)
            
        return indices

    def decode(self, indices, remove_special_tokens=True):
        """Convert token indices back to text."""
        chars = []
        
        for idx in indices:
            char = self.idx_to_char.get(idx, self.special_tokens['UNK'])
            if remove_special_tokens and char in self.special_tokens.values():
                continue
            chars.append(char)
            
        return ''.join(chars)
    
    def batch_encode(self, texts, max_length=None, padding=True):
        """Encode a batch of texts."""
        encoded = [self.encode(text) for text in texts]
        
        if max_length is None and padding:
            max_length = max(len(seq) for seq in encoded)
            
        if padding:
            # Pad sequences to max_length
            encoded = [
                seq + [self.pad_idx] * (max_length - len(seq))
                for seq in encoded
            ]
            
        return encoded
    
    def save_vocab(self, path):
        """Save vocabulary to file."""
        vocab_data = {
            'char_to_idx': self.char_to_idx,
            'special_tokens': self.special_tokens
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    @classmethod
    def load_vocab(cls, path):
        """Load vocabulary from file."""
        with open(path) as f:
            vocab_data = json.load(f)
            
        tokenizer = cls()
        tokenizer.char_to_idx = vocab_data['char_to_idx']
        tokenizer.special_tokens = vocab_data['special_tokens']
        tokenizer.idx_to_char = {idx: char for char, idx in tokenizer.char_to_idx.items()}
        tokenizer.vocab_size = len(tokenizer.char_to_idx)
        
        return tokenizer


def get_batch(data, batch_size, block_size, device="cpu"):
    """Generate a small batch of data for training"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data[i : i + block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i + 1 : i + block_size + 1]) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def train_model(
    model, train_data, config, epochs=10, batch_size=32, learning_rate=3e-4
):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Create progress bar for epochs
    pbar = tqdm(range(epochs), desc="Training")

    losses = []
    for epoch in pbar:
        # Get random batch and compute loss
        X, Y = get_batch(train_data, batch_size, config.block_size)
        logits, loss = model(X, Y)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return losses


def generate_text(model, tokenizer, start_text, max_new_tokens=50, temperature=0.7):
    model.eval()
    context = torch.tensor(tokenizer.encode(start_text), dtype=torch.long).unsqueeze(0)
    
    def sample(logits, top_k=40):
        # Filter out bad tokens
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float('-inf')
        
        # Apply temperature
        probs = F.softmax(logits / temperature, dim=-1)
        
        # Sample from the filtered distribution
        return torch.multinomial(probs, num_samples=1)
    
    generated = model.generate(
        context, 
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        sample_fn=sample
    )
    
    return tokenizer.decode(generated[0].tolist())


# Fun training data - a mix of movie quotes!
movie_quotes = """
To infinity and beyond!
I'll be back.
May the Force be with you.
Life is like a box of chocolates.
Here's looking at you, kid.
There's no place like home.
I am your father. Or your mother.
E.T. phone home. Or not.
I see dead people and I'm not afraid.
You're gonna need a bigger boat.
Elementary, my dear Watson.
I'll have what she's having.
You can't handle the truth!
Houston, we have a problem.
Do, or do not. There is no try.
I feel the need... the need for speed!
They may take our lives, but they'll never take our freedom!
Why so serious?
I'm king of the world!
Hasta la vista, baby.
My name is Bond, James Bond.
I'm going to make him an offer he can't refuse.
You're gonna need a bigger boat.
Let's put a smile on that face.
I'm the king of the world!
What's the matter with you people?
I'm not even supposed to be here today.
Give me a break! Give peace a chance.
All right, Mr. DeMille, I'm ready for my close-up.
C'mon, let's go bowling!
Big Lebowski was a great movie.
Ich bin ein Berliner.
Dude, where's my car?
Positively fourth street.
A little bit of South Philly never hurt nobody.
"""

if __name__ == "__main__":
    # Create and fit tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.fit(movie_quotes)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Print some stats
    print("\nSample tokens:")
    sample_tokens = list(tokenizer.char_to_idx.keys())[:20]
    for token in sample_tokens:
        print(f"'{token}': {tokenizer.char_to_idx[token]}")

    # Test encoding/decoding
    test_text = "Life is like a box of chocolates"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Convert text to tokens with special tokens
    data = tokenizer.encode(movie_quotes)

    # Optional: Save vocabulary for later use
    tokenizer.save_vocab('tokenizer_vocab.json')

    # Create model config
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=64,
        n_layer=6,
        n_embd=256,
        num_heads=8,
        head_size=32,
        dropout=0.2
    )

    # Create model
    model = MiniGPT(config)
    print("Training model...")

    # Train model
    losses = train_model(model, data, config, epochs=500, batch_size=8)

    # Generate some text!
    print("\nGenerating text...\n")
    prompts = ["I am", "Life is", "May the", "To infinity", "My name is"]

    for prompt in prompts:
        generated = generate_text(model, tokenizer, prompt, max_new_tokens=50)
        print(f"Prompt: '{prompt}'")
        print(f"Generated (clean): {generated}")
