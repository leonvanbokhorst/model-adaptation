"""
Long Short-Term Memory (LSTM) Networks and Their Significance

LSTMs were introduced in 1997 by Hochreiter & Schmidhuber to solve the vanishing gradient
problem in traditional RNNs. They're particularly good at learning long-term dependencies
in sequential data.

Key Components of an LSTM:
1. Forget Gate: Decides what information to throw away from the cell state
2. Input Gate: Decides which new information to store in the cell state
3. Candidate Memory: Creates new candidate values that could be added to the state
4. Output Gate: Decides what parts of the cell state to output

The LSTM's power comes from its cell state (C_t), which acts like a conveyor belt.
Information can flow along it unchanged, and the network can learn to add or remove
information from the cell state, regulated by the gates.

The gates are the key innovation:
- They use sigmoid functions that output numbers between 0 and 1
- These numbers are used as filters (0 = "let nothing through", 1 = "let everything through")
- The network learns what information is important to keep or throw away

Mathematical Formulation:
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     # Input gate
C̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)  # Candidate memory
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     # Output gate
C_t = f_t * C_{t-1} + i_t * C̃_t         # Cell state update
h_t = o_t * tanh(C_t)                   # Hidden state update

Where:
- σ is the sigmoid function
- * is element-wise multiplication
- [h_{t-1}, x_t] is concatenation of previous hidden state and current input
"""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM implementation."""

    def __init__(self, input_size, hidden_size):
        """
        Initialize LSTM components.
        
        Args:
            input_size: Dimension of input features
            hidden_size: Dimension of hidden state and cell state
        """
        super().__init__()
        self.hidden_size = hidden_size

        # Each gate is a linear transformation followed by a sigmoid
        # They take concatenated (previous_hidden, current_input) as input
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate_memory = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()  # For gates (0-1 range)
        self.tanh = nn.Tanh()       # For state updates (-1 to 1 range)

    def forward(self, x, state=None):
        """
        Process one timestep through the LSTM.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            state: Tuple of (hidden_state, cell_state) or None for initial step
        
        Returns:
            tuple: (new_hidden_state, (new_hidden_state, new_cell_state))
        """
        batch_size = x.size(0)

        # Initialize states to zeros if not provided
        if state is None:
            h_prev = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_prev = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_prev, c_prev = state

        # Combine input and previous hidden state for gate calculations
        combined = torch.cat((x, h_prev), dim=1)

        # Gate calculations
        forget = self.sigmoid(self.forget_gate(combined))
        input = self.sigmoid(self.input_gate(combined))
        candidate = self.tanh(self.candidate_memory(combined))
        output = self.sigmoid(self.output_gate(combined))

        # State updates
        c_next = forget * c_prev + input * candidate  # Update cell state
        h_next = output * self.tanh(c_next)          # Create hidden state

        return h_next, (h_next, c_next)

    def get_gate_states(self, x, state=None):
        """Returns internal gate states for visualization."""
        batch_size = x.size(0)
        
        if state is None:
            h_prev = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_prev = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_prev, c_prev = state
            
        combined = torch.cat((x, h_prev), dim=1)
        
        # Get all gate states
        forget = self.sigmoid(self.forget_gate(combined))
        input = self.sigmoid(self.input_gate(combined))
        candidate = self.tanh(self.candidate_memory(combined))
        output = self.sigmoid(self.output_gate(combined))
        
        # Update states
        c_next = forget * c_prev + input * candidate
        h_next = output * self.tanh(c_next)
        
        return {
            'forget_gate': forget,
            'input_gate': input,
            'candidate_memory': candidate,
            'output_gate': output,
            'cell_state': c_next,
            'hidden_state': h_next
        }


# Example usage with a simple sequence task
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = LSTM(input_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x_sequence):
        outputs = []
        h = None

        # Process each timestep
        for t in range(x_sequence.size(1)):
            x_t = x_sequence[:, t, :]
            h_t, h = self.lstm(x_t, h)
            out_t = self.output_layer(h_t)
            outputs.append(out_t)

        return torch.stack(outputs, dim=1)


def test_lstm_model(model, test_sequences, test_targets):
    """
    Test the LSTM model and visualize its predictions.
    
    Args:
        model: Trained LSTMPredictor model
        test_sequences: Input sequences of shape (batch_size, seq_length, input_size)
        test_targets: Target values of shape (batch_size, seq_length, 1)
    
    Returns:
        dict: Dictionary containing test metrics
    """
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        # Get predictions
        predictions = model(test_sequences)

        # Calculate test loss
        test_loss = nn.MSELoss()(predictions, test_targets)

        # Calculate metrics
        mae = torch.mean(torch.abs(predictions - test_targets))
        mse = torch.mean((predictions - test_targets) ** 2)
        rmse = torch.sqrt(mse)

        return {
            'test_loss': test_loss.item(),
            'mae': mae.item(),
            'mse': mse.item(),
            'rmse': rmse.item(),
        }

def demo_lstm():
    # Create synthetic sequence data
    seq_length = 10
    batch_size = 32
    input_size = 5
    hidden_size = 10
    output_size = 1

    # Create model
    model = LSTMPredictor(input_size, hidden_size, output_size)

    # Generate random sequences
    x = torch.randn(batch_size, seq_length, input_size)
    # Target: sum of input features at each timestep
    y = torch.sum(x, dim=2, keepdim=True)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print("Training LSTM...")
    for epoch in range(1000):
        optimizer.zero_grad()

        # Forward pass
        y_pred = model(x)
        loss = criterion(y_pred, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}")

    # Generate test data
    test_x = torch.randn(batch_size, seq_length, input_size)
    test_y = torch.sum(test_x, dim=2, keepdim=True)
    
    # Test the model
    print("\nTesting LSTM...")
    metrics = test_lstm_model(model, test_x, test_y)
    
    # Print metrics
    print("\nTest Results:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    demo_lstm()
