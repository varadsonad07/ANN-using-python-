# Relu activation function with gradient descent

import numpy as np

# ReLU Activation Function
def ReLU(x):
    return np.maximum(0, x)

# Derivative of ReLU
def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)

# Data (Simple Example)
x = np.array([1, 2, 3, 4, 5])  # Input
y = np.array([1, 2, 3, 4, 5])  # Target (ideal output)

# Initialize weight
weight = np.random.randn()

# Hyperparameters
lr = 0.01
epochs = 100

# Training loop
for epoch in range(epochs):
    # Forward pass
    z = x * weight
    y_pred = ReLU(z)

    # Calculate loss (Mean Squared Error)
    loss = np.mean((y - y_pred) ** 2)

    # Backpropagation
    grad = ReLU_derivative(z)
    weight -= lr * np.sum((y_pred - y) * x * grad)  # Weight update

    # Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss={loss:.4f}, Weight={weight:.4f}")

# Final result
print("\nFinal Weight:", weight)
print("Final Predictions:", np.round(ReLU(x * weight), 2))
