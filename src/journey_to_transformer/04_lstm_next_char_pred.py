import torch
import torch.nn as nn
import string


class TextPredictor(nn.Module):
    """
    Neural network for predicting the next character in a sequence.
    Uses LSTM (Long Short-Term Memory) architecture for understanding patterns in text.
    """
    def __init__(self, vocab_size, embedding_dim=32, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size

        # Embedding layer: converts character indices to dense vectors
        # - Each character gets a learned vector representation
        # - Similar to word embeddings but for individual characters
        # - embedding_dim controls how detailed these representations are
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer: processes sequences and maintains memory
        # - input_size: size of embedded character vectors
        # - hidden_size: how much information to remember
        # - num_layers=2: stacked LSTMs for more complex patterns
        # - batch_first=True: expect data in (batch, sequence, features) format
        # - dropout=0.2: randomly drop 20% of connections for regularization
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        # Final layer: convert LSTM output to character probabilities
        # - Takes LSTM's hidden state
        # - Outputs scores for each possible character
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # 1. Convert character indices to embeddings
        embeds = self.embedding(x)

        # 2. Process sequence through LSTM
        # - Returns processed sequence and updated hidden state
        # - hidden state carries memory between batches
        lstm_out, hidden = self.lstm(embeds, hidden)

        # 3. Convert LSTM output to character predictions
        output = self.fc(lstm_out)
        return output, hidden


# Text processing utilities
class TextProcessor:
    """
    Handles conversion between text and the numerical format needed by the network.
    Think of it as a translator between human-readable text and network-readable numbers.
    """
    def __init__(self):
        # Create character mappings using all printable ASCII characters
        # - Includes letters, numbers, punctuation, and whitespace
        # - char_to_idx: converts characters to unique numbers
        # - idx_to_char: converts numbers back to characters
        self.chars = string.printable
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = dict(enumerate(self.chars))
        self.vocab_size = len(self.chars)

    def encode(self, text):
        """Convert text string to tensor of indices."""
        return torch.tensor([self.char_to_idx[ch] for ch in text])

    def decode(self, indices):
        """Convert tensor of indices back to text string."""
        return "".join([self.idx_to_char[idx.item()] for idx in indices])


def generate_text(model, processor, start_text="Hello", length=100, temperature=0.8):
    """
    Generate new text by sampling from the model's predictions.
    
    Parameters:
    - start_text: initial text to seed the generation
    - length: how many characters to generate
    - temperature: controls randomness of sampling
        - Lower (e.g., 0.5): more conservative, predictable text
        - Higher (e.g., 1.2): more creative, potentially chaotic text
    """
    model.eval()  # Switch to evaluation mode
    current_text = start_text
    hidden = None  # LSTM's memory state

    with torch.no_grad():  # Don't track gradients during generation
        for _ in range(length):
            # 1. Prepare input sequence
            x = processor.encode(current_text)
            x = x.unsqueeze(0)  # Add batch dimension

            # 2. Get model's predictions
            output, hidden = model(x, hidden)

            # 3. Apply temperature to adjust prediction randomness
            # - Higher temperature = more uniform probabilities
            # - Lower temperature = more peaked probabilities
            probs = torch.softmax(output[0, -1] / temperature, dim=0)

            # 4. Sample next character from probability distribution
            next_char_idx = torch.multinomial(probs, 1)
            next_char = processor.decode([next_char_idx])

            # 5. Add to generated text
            current_text += next_char

    return current_text


def train_model():
    # Sample training text (you can replace this with your own text)
    text = """The quick brown fox jumps over the lazy dog.
    Smalltalk is a fantastic programming language.
    LSTMs are great for processing sequential data.
    Neural networks learn from examples."""

    # Setup
    processor = TextProcessor()
    model = TextPredictor(processor.vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Prepare data
    sequence_length = 20
    sequences = []
    targets = []

    # Create training sequences
    for i in range(len(text) - sequence_length):
        sequences.append(text[i : i + sequence_length])
        targets.append(text[i + sequence_length])

    print("Training the model...")
    for epoch in range(100):
        model.train()
        total_loss = 0

        for seq, target in zip(sequences, targets):
            # Prepare data
            x = processor.encode(seq).unsqueeze(0)  # Shape: [1, seq_len]
            y = processor.encode(target)  # Shape: [1]

            # Forward pass
            output, _ = model(x)  # output shape: [1, seq_len, vocab_size]

            # Get only the last prediction and reshape
            last_output = output[:, -1, :]  # Shape: [1, vocab_size]

            # Loss calculation
            loss = criterion(last_output, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {total_loss/len(sequences):.4f}")
            # Generate sample text
            sample = generate_text(model, processor, "The quick brown ", length=50)
            print(f"Sample text: {sample}\n")

    return model, processor


if __name__ == "__main__":
    # Train the model
    model, processor = train_model()

    # Generate some text
    print("\nGenerating text with different temperatures:")
    for temp in [0.5, 0.8, 1.2]:
        print(f"\nTemperature: {temp}")
        generated = generate_text(
            model, processor, "The quick brown ", length=100, temperature=temp
        )
        print(generated)
