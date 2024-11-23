import torch
import torch.nn as nn
import matplotlib.pyplot as plt


"""
RNN Mood Predictor: Understanding Sequential Data and Memory

This example demonstrates how Recurrent Neural Networks (RNNs) can process sequences
of events and maintain a "memory" of past events to make predictions. It's like how
your mood throughout the day is influenced by the sequence of events you experience.

Key Concepts:
1. Sequential Processing - RNNs handle data that comes in sequences (like events in a day)
2. Hidden State - The network maintains a "memory" of previous events
3. Time Steps - Each event is processed one at a time, updating the memory
4. Non-linear Transformations - Using activation functions to model complex patterns

Historical Significance:
- RNNs were a breakthrough in handling sequential data
- They enabled applications like:
  * Natural language processing
  * Time series prediction
  * Music generation
  * Speech recognition

The network uses:
- Input layer: Transforms each event into a hidden representation
- RNN cell: Updates the memory based on current event and previous state
- Output layer: Makes predictions based on current memory state
"""

# Our Simple RNN - like a friend who remembers your day's events!
class MoodPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size  # size of the memory

        # Transform input events
        self.input_layer = nn.Linear(
            input_size, hidden_size
        )  # used to transform the input events

        # The "memory" layer - remembers previous events
        self.rnn_cell = nn.RNNCell(
            hidden_size, hidden_size
        )  # used to update the memory

        # Final prediction layer
        self.output_layer = nn.Linear(
            hidden_size, output_size
        )  # used to make a prediction

        # Activation functions
        self.tanh = nn.Tanh()  # used to squash the values between -1 and 1
        self.sigmoid = nn.Sigmoid()  # used to squash the values between 0 and 1

    def forward(self, x, hidden=None):
        # For first event of day, start with neutral state
        if hidden is None:
            hidden = torch.zeros(x.size(0), self.hidden_size).to(x.device)

        # Lists to store predictions at each time step
        outputs = []

        # Process each event in the sequence
        for t in range(x.size(1)):
            # Get current event
            current_input = x[:, t, :]

            # Transform input
            transformed = self.tanh(self.input_layer(current_input))

            # Update memory with new event
            hidden = self.rnn_cell(transformed, hidden)

            # Make prediction
            output = self.sigmoid(self.output_layer(hidden))
            outputs.append(output)

        # Stack all predictions together
        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden


# Let's create some example data!
def generate_day_sequences(num_sequences=100):
    """
    Generate synthetic day sequences to train our model.
    
    Each day is represented as a sequence of 5 events:
    - Events are one-hot encoded: [good, neutral, bad]
    - Final mood is calculated based on the balance of good vs bad events
    - Some randomness is added to make it more realistic
    
    This is like how your actual day might have a mix of events that
    collectively influence your final mood.
    """
    sequences = []
    labels = []

    for _ in range(num_sequences):
        # Generate random day sequence
        day = torch.zeros(5, 3)
        for t in range(5):
            # Random event type (one-hot encoded)
            event_type = torch.randint(0, 3, (1,))
            day[t, event_type] = 1

        # Calculate mood based on events (with some randomness)
        good_events = day[:, 0].sum()
        bad_events = day[:, 2].sum()
        mood = torch.sigmoid(torch.tensor([(good_events - bad_events) / 2]))

        sequences.append(day)
        labels.append(mood)

    return torch.stack(sequences), torch.stack(labels)


# Training time!
def train_and_test():
    """
    Train the mood predictor and evaluate its performance.
    
    The training process:
    1. Split data into training and test sets
    2. Train model for 100 epochs
    3. Use Binary Cross Entropy loss (good for 0-1 predictions)
    4. Use Adam optimizer (adaptive learning rates)
    5. Evaluate on test set
    6. Visualize training progress
    
    This mimics how we might train a real mood prediction system,
    though real-world data would be much more complex!
    """
    # Generate data
    X, y = generate_day_sequences()

    # Split into train and test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create model
    model = MoodPredictor(input_size=3, hidden_size=12, output_size=1)

    # Training setup
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    print("Training the mood predictor...")
    losses = []

    for epoch in range(100):
        optimizer.zero_grad()

        # Forward pass
        outputs, _ = model(X_train)
        loss = criterion(outputs[:, -1], y_train)  # Only care about final prediction

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

    # Test the model
    model.eval()
    with torch.no_grad():
        _calculate_test_loss_and_accuracy(model, X_test, criterion, y_test)
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    return model


def _calculate_test_loss_and_accuracy(model, X_test, criterion, y_test):
    test_outputs, _ = model(X_test)
    test_predictions = test_outputs[:, -1]
    test_loss = criterion(test_predictions, y_test)

    # Convert predictions to binary decisions with a threshold of 0.5
    binary_preds = (test_predictions >= 0.5).float()
    binary_targets = (y_test >= 0.5).float()
    accuracy = (binary_preds == binary_targets).float().mean()

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.2%}")


# Let's run it!
if __name__ == "__main__":
    model = train_and_test()

    # Try a specific day sequence
    good_day = torch.tensor(
        [
            [1, 0, 0],  # Good morning
            [1, 0, 0],  # Nice lunch
            [0, 1, 0],  # Normal afternoon
            [0, 0, 1],  # Minor setback
            [1, 0, 0],  # Great evening
        ],
        dtype=torch.float32,  # Specify float32 data type
    ).unsqueeze(0)

    bad_day = torch.tensor(
        [
            [0, 0, 1],  # Bad morning
            [0, 1, 0],  # Meh lunch
            [0, 0, 1],  # Bad afternoon
            [0, 0, 1],  # Bad evening
            [1, 0, 0],  # Good night
        ],
        dtype=torch.float32,  # Add float32 data type
    ).unsqueeze(
        0
    )  # Add batch dimension

    with torch.no_grad():
        predictions, _ = model(good_day)
        final_mood = predictions[0, -1].item()
    print(f"\nPredicted mood for the good day: {final_mood:.2%}")

    with torch.no_grad():
        predictions, _ = model(bad_day)
        final_mood = predictions[0, -1].item()
    print(f"Predicted mood for the bad day: {final_mood:.2%}")
