"""
The XOR Problem and Its Historical Significance

The XOR (exclusive OR) problem was a pivotal challenge in AI history that helped lead to 
the first AI winter in the 1970s. The controversy began when Marvin Minsky and Seymour 
Papert published their 1969 book "Perceptrons", which demonstrated that single-layer 
perceptrons could not solve the XOR problem.

The XOR function returns:
- 1 when inputs are different (1,0) or (0,1)
- 0 when inputs are same (0,0) or (1,1)

This created a crisis because:
1. XOR is a simple logical operation that humans can easily understand
2. Single-layer perceptrons could not learn this pattern
3. It wasn't clear if adding layers would help or if they could be trained effectively

The solution emerged in the 1980s with:
1. Multi-layer networks (adding hidden layers)
2. Backpropagation algorithm for training
3. Non-linear activation functions

This combination allowed neural networks to learn the complex decision boundaries needed
for XOR, helping to end the first AI winter. The XOR problem demonstrates that:
- Sometimes simple-looking problems require complex solutions
- The limitations of one approach can drive innovation in new directions
- Understanding failure cases is crucial for advancing the field

The network below uses:
- 2 input neurons (for the two binary inputs)
- 4 hidden neurons (to create complex decision boundaries)
- 1 output neuron (for the binary output)
- ReLU activation (to introduce non-linearity)


It doesn't always learn correctly. This is a classic case of the network getting stuck in a 
local minimum - in this case, it's actually stuck at its initial state where it's just 
predicting 0.5 for everything. The constant loss of 0.6931 (which is approximately -ln(0.5)) 
is a telltale sign that the network isn't learning at all.

This happens because:
- Neural networks are initialized with random weights
- Sometimes these initial weights lead to a configuration where the gradients aren't 
  strong enough to push the network out of this "lazy" state
- The network finds it's "comfortable" just predicting 0.5 for everything, as this 
  minimizes its maximum error for any input

Solutions typically include:
- Just restart training with new random weights (reinitialize the model)
- Try different learning rates
- Use different weight initialization strategies
- Add momentum to the optimizer

This is actually a great learning example because it shows how neural networks can sometimes 
get stuck, just like humans can get stuck in suboptimal thinking patterns! The good news is 
that if you just run the code again, the new random initialization will likely give you 
better results.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Configure matplotlib to work in VS Code
plt.switch_backend('TkAgg')

class XORNetwork(nn.Module):
    """
    A simple neural network for solving the XOR problem.
    """
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # First layer: 2 inputs -> 4 neurons
            # We need 4 neurons because XOR is a complex pattern:
            # - 2 neurons aren't enough to separate the data properly
            # - 4 neurons give us more "decision boundaries" to work with
            nn.Linear(2, 4),
            
            # ReLU activation function
            # - Converts negative numbers to 0
            # - Keeps positive numbers as they are
            # - Helps network learn non-linear patterns
            nn.ReLU(),
            
            # Output layer: 4 neurons -> 1 output
            # - Takes the 4 intermediate values
            # - Combines them into final yes/no decision
            nn.Linear(4, 1),
            
            # Sigmoid squishes output between 0 and 1
            # - Perfect for yes/no decisions
            # - 0 = false, 1 = true
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)

# Create training data
# XOR truth table: output is 1 if inputs are different, 0 if same
X = torch.tensor([[0.0, 0.0],  # Input: (0,0) -> Output should be 0
                 [0.0, 1.0],   # Input: (0,1) -> Output should be 1
                 [1.0, 0.0],   # Input: (1,0) -> Output should be 1
                 [1.0, 1.0]])  # Input: (1,1) -> Output should be 0

y = torch.tensor([[0.0],  # Expected output for (0,0)
                 [1.0],   # Expected output for (0,1)
                 [1.0],   # Expected output for (1,0)
                 [0.0]])  # Expected output for (1,1)

# Create network and training tools
model = XORNetwork()
# Binary Cross Entropy Loss: good for yes/no problems
criterion = nn.BCELoss()
# Adam optimizer: automatically adjusts learning speed
# lr=0.05 means "take bigger steps" when learning
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# Keep track of how well we're learning
losses = []

print("Training the network to solve XOR...")
print("Epoch   Loss")
print("-" * 20)

# Train for 1000 rounds
for epoch in range(1000):
    # 1. Make a prediction with current network
    output = model(X)
    # 2. Calculate how wrong we were
    loss = criterion(output, y)
    # 3. Reset gradients from last time
    optimizer.zero_grad()
    # 4. Calculate how to adjust the network
    loss.backward()
    # 5. Update the network
    optimizer.step()
    
    # Store loss for plotting
    losses.append(loss.item())
    
    # Show progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"{epoch+1:5d}   {loss.item():.4f}")

# Test how well we learned
print("\nTesting the network:")
print("Input  Target  Prediction  Result")
print("-" * 40)
with torch.no_grad():  # Don't need gradients for testing
    predictions = model(X)
    for i in range(len(X)):
        prediction = predictions[i].item()
        target = y[i].item()
        # Consider prediction wrong if it's more than 0.2 away from target
        is_correct = abs(prediction - target) < 0.2
        result = "✅" if is_correct else "💥"
        print(f"{X[i].numpy()}  {target:.0f}       {prediction:.3f}     {result}")

# Plot how the learning progressed
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title("Training Loss Over Time")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

print("\nLook how quickly it learns! Much faster than waiting 17 years... 😉")
