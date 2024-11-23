"""
Understanding Softmax: The Neural Network's Decision Maker

Softmax is a crucial function in neural networks that converts raw scores (logits) into 
probabilities. It's used when we want our network to make decisions between multiple options.

Key Properties of Softmax:
1. Converts any real numbers into probabilities (0-1)
2. Ensures all outputs sum to 1.0
3. Maintains relative differences (bigger inputs = bigger probabilities)
4. Handles both positive and negative numbers

Historical Significance:
- Introduced in 1959 by R. Duncan Luce in "Individual Choice Behavior"
- Became fundamental in neural networks during the 1980s
- Critical for modern classification tasks

Why We Need Softmax:
- Raw neural network outputs can be any number
- We often need probabilities for decision making
- Helps with training stability
- Makes outputs interpretable
"""

import torch
import torch.nn as nn

# Raw scores for where to get lunch
scores = torch.tensor([10.0, 2.0, 5.0])  # Pizza, Salad, Tacos
print("Raw scores:", scores)

def softmax(x):
    """
    Converts raw scores into probabilities using the softmax function:
    P(i) = exp(x[i]) / sum(exp(x))
    
    Why exp()?
    - Always positive (we can't have negative probabilities)
    - Maintains relative differences
    - Differentiable (important for training)
    """
    exp_x = torch.exp(x)  # Step 1: Convert to positive numbers
    return exp_x / exp_x.sum()  # Step 2: Normalize to sum to 1

# Apply softmax to our lunch scores
probabilities = softmax(scores)
print("\nAfter softmax (probabilities):", probabilities)
print("Notice they sum to 1:", probabilities.sum())

class SimpleClassifier(nn.Module):
    """
    A basic neural network classifier that demonstrates softmax in action.
    
    Architecture:
    - Input layer (2 features)
    - Single linear layer
    - Output layer (3 classes)
    """
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 3)  # 2 inputs, 3 classes
        # Note: PyTorch's CrossEntropyLoss includes softmax!

    def forward(self, x):
        # Raw scores (logits)
        scores = self.layer(x)

        # Compare outputs before and after softmax
        raw_output = scores  # Raw network outputs (can be any number)
        probability_output = softmax(scores)  # Converted to probabilities

        return raw_output, probability_output

# Demonstrate with real data
model = SimpleClassifier()
# Two samples with two features each
input_data = torch.tensor([[2.0, 1.0], [1.0, 3.0]])  

raw, probs = model(input_data)
print("\nRaw network outputs (can be any number):")
print(raw)
print("\nAfter softmax (nice probabilities between 0-1):")
for i in range(len(probs)):
    print(f"Sample {i+1}: {probs[i]}")
print("\nEach row sums to:", probs.sum(dim=1))  # Always 1!

# Demonstrate how softmax handles different scenarios
print("\nScenario 1: Similar inputs")
small_diffs = torch.tensor([2.0, 2.1, 2.2])
print("Input:", small_diffs)
print("Output (notice gentle preferences):", softmax(small_diffs))

print("\nScenario 2: Very different inputs")
big_diffs = torch.tensor([2.0, 4.0, 2.2])
print("Input:", big_diffs)
print("Output (notice strong preference):", softmax(big_diffs))

print("\nScenario 3: Mixed positive/negative")
mixed_numbers = torch.tensor([-1.0, 5.0, 2.0])
print("Input:", mixed_numbers)
print("Output (still works!):", softmax(mixed_numbers))  
