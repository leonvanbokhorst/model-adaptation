import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(x.size(0), self.hidden_size)

        outputs = []
        for t in range(x.size(1)):
            hidden = self.rnn_cell(x[:, t, :], hidden)
            output = self.output(hidden)
            outputs.append(output)
        return torch.stack(outputs, 1)


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.output(lstm_out)


def create_tricky_memory_test(sequence_length=100, batch_size=32):
    """
    Creates a MUCH harder memory test:
    - Longer sequences (100 timesteps)
    - Multiple important events to remember
    - Random noise to distract the network
    - Multiple sequences at once (batch_size)
    """
    # Initialize input and target sequences
    x = torch.zeros(batch_size, sequence_length, 5)  # 5 input features now!
    y = torch.zeros(batch_size, sequence_length, 1)

    for b in range(batch_size):
        # Place important events (1s) at random positions in first channel
        important_positions = torch.randint(0, sequence_length // 2, (2,))
        x[b, important_positions, 0] = 1

        # Add random noise in other channels
        x[b, :, 1:] = torch.randn(sequence_length, 4) * 0.5

        # Target: Remember the important events forever
        for pos in important_positions:
            y[b, pos:, 0] = 1

    return x, y


# Training function with visualization
def train_and_compare(sequence_length=100, hidden_size=32, epochs=200):
    # Create models
    rnn_model = SimpleRNN(input_size=5, hidden_size=hidden_size, output_size=1)
    lstm_model = SimpleLSTM(input_size=5, hidden_size=hidden_size, output_size=1)

    # Training setup
    criterion = nn.BCEWithLogitsLoss()
    rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.01)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)

    # Training history
    rnn_losses = []
    lstm_losses = []

    print("Training both models...")
    for epoch in range(epochs):
        # Generate new random sequences each epoch
        x, y = create_tricky_memory_test(sequence_length)

        # Train RNN
        rnn_optimizer.zero_grad()
        rnn_out = rnn_model(x)
        rnn_loss = criterion(rnn_out, y)
        rnn_loss.backward()
        rnn_optimizer.step()
        rnn_losses.append(rnn_loss.item())

        # Train LSTM
        lstm_optimizer.zero_grad()
        lstm_out = lstm_model(x)
        lstm_loss = criterion(lstm_out, y)
        lstm_loss.backward()
        lstm_optimizer.step()
        lstm_losses.append(lstm_loss.item())

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"RNN Loss: {rnn_loss.item():.4f}")
            print(f"LSTM Loss: {lstm_loss.item():.4f}\n")

    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(rnn_losses, label="RNN")
    plt.plot(lstm_losses, label="LSTM")
    plt.title("Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Test with a single sequence for visualization
    x_test, y_test = create_tricky_memory_test(sequence_length, batch_size=1)

    with torch.no_grad():
        rnn_test = torch.sigmoid(rnn_model(x_test))
        lstm_test = torch.sigmoid(lstm_model(x_test))

    # Plot test sequence predictions
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x_test[0, :, 0].numpy(), label="Important Events", marker="o")
    plt.plot(rnn_test[0, :, 0].numpy(), label="RNN Prediction", alpha=0.7)
    plt.title("RNN Memory Test")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x_test[0, :, 0].numpy(), label="Important Events", marker="o")
    plt.plot(lstm_test[0, :, 0].numpy(), label="LSTM Prediction", alpha=0.7)
    plt.title("LSTM Memory Test")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Run the comparison!
train_and_compare()
