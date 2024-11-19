"""
Neural Network Evolution: From Basic to Modern Architectures

This file demonstrates the historical evolution of neural network architectures,
showing how various improvements helped solve fundamental problems:

1. BasicNetwork: Uses sigmoid activation (historical approach from 1980s)
2. ImprovedNetwork: Uses tanh activation (1990s improvement)
3. ModernNetwork: Implements batch normalization and ReLU (2010s best practices)
4. SimpleMemoryNetwork: Demonstrates early memory concepts (precursor to LSTM)

Each network shows key innovations that helped advance deep learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# 1. Basic Network with Sigmoid (prone to vanishing gradients)
class BasicNetwork(nn.Module):
    """
    Represents the earliest practical neural networks (1980s-style).
    
    Problems with this architecture:
    - Sigmoid activation suffers from vanishing gradients
    - Gradients become very small for extreme values
    - Network learns very slowly in deep layers
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()  # Historical activation function
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.sigmoid(self.layer1(x))  # Sigmoid squashes values to (0,1)
        x = self.layer2(x)
        return x


# 2. Improved Network with Better Activation
class ImprovedNetwork(nn.Module):
    """
    Represents 1990s improvements with tanh activation.
    
    Advantages over sigmoid:
    - Outputs centered around 0 (-1 to 1 range)
    - Stronger gradients
    - Generally faster convergence
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.tanh = nn.Tanh()  # Centered activation function
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.layer2(x)
        return x


# 3. Deep Network with Modern Solutions
class ModernNetwork(nn.Module):
    """
    Represents current best practices (2010s onwards).
    
    Key modern features:
    - ReLU activation (solves vanishing gradient)
    - Batch Normalization (stabilizes training)
    - Deeper architecture (more layers)
    - Xavier/Glorot initialization (built into PyTorch)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Modern architecture with multiple improvements
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)  # Normalizes layer outputs
        self.relu = nn.ReLU()  # Modern activation function
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Each layer follows the pattern: Linear -> BatchNorm -> ReLU
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.layer3(x)  # No activation on final layer
        return x


# 4. Early LSTM-like Memory (simplified for demonstration)
class SimpleMemoryNetwork(nn.Module):
    """
    Demonstrates early attempts at networks with memory (pre-LSTM).
    
    Key concepts:
    - Input gate: Controls what information to store
    - Memory cell: Maintains state over time
    - Output gate: Controls what information to output
    
    This is a simplified version showing the concept that led to LSTM/GRU.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Gates control information flow
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.memory_transform = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        
        self.output = nn.Linear(hidden_size, output_size)
        
        # Activation functions for different purposes
        self.tanh = nn.Tanh()  # For memory content
        self.sigmoid = nn.Sigmoid()  # For gates

    def forward(self, x, hidden_state=None):
        batch_size = x.size(0)
        
        # Initialize hidden state if none provided
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, self.hidden_size).to(x.device)

        # Combine current input with previous state
        combined = torch.cat((x, hidden_state), dim=1)
        
        # Gate mechanisms
        input_gate = self.sigmoid(self.input_gate(combined))
        memory_write = self.tanh(self.memory_transform(combined))
        output_gate = self.sigmoid(self.output_gate(combined))
        
        # Update memory state
        memory_cell = input_gate * memory_write
        
        # Generate output using gated memory
        hidden_state = output_gate * self.tanh(memory_cell)
        output = self.output(hidden_state)

        return output, hidden_state


# Demonstration
def train_and_compare():
    """
    Trains all network variants on a simple task and compares their performance.
    
    The task is to sum input features - chosen because:
    - It's simple enough to learn quickly
    - Complex enough to show differences between architectures
    - Easy to verify results
    """
    # Generate some sample data
    X = torch.randn(100, 10)  # 100 samples, 10 features
    y = torch.sum(X, dim=1).unsqueeze(1)  # Simple sum task

    # Create networks
    networks = {
        "Basic (Sigmoid)": BasicNetwork(10, 20, 1),
        "Improved (Tanh)": ImprovedNetwork(10, 20, 1),
        "Modern (ReLU+BN)": ModernNetwork(10, 20, 1),
        "Memory Net": SimpleMemoryNetwork(10, 20, 1),
    }

    # Training settings
    epochs = 500
    losses = {name: [] for name in networks}

    for name, net in networks.items():
        print(f"\nTraining {name}...")
        optimizer = optim.Adam(net.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Forward pass (handle memory network separately)
            if isinstance(net, SimpleMemoryNetwork):
                output, _ = net(X)
            else:
                output = net(X)

            # Compute loss
            loss = criterion(output, y)
            losses[name].append(loss.item())

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    return losses


# Run training and plot results
losses = train_and_compare()

plt.figure(figsize=(10, 6))
for name, loss_values in losses.items():
    plt.plot(loss_values, label=name)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.yscale("log")  # Better visualization of loss differences
plt.grid(True)
plt.show()
