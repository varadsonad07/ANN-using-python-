#Implementation of Activation Functions: Sigmoid,ReLU, and Tanh in Neural Networks
#2. Temperature Prediction
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Step 2: Generate synthetic dataset
np.random.seed(0)
altitude = np.random.uniform(0, 3000, 100)  # Altitude in meters
temperature = 25 - (altitude / 200) + np.random.normal(0, 2, 100)  # Temperature in °C

# Normalize data
altitude = (altitude - np.mean(altitude)) / np.std(altitude)
temperature = (temperature - np.mean(temperature)) / np.std(temperature)

# Step 3: Initialize parameters
w = np.random.randn()
b = np.random.randn()
learning_rate = 0.01
epochs = 1000

# Step 4: Gradient Descent function with activation function selection
def gradient_descent(w, b, learning_rate, activation_func, epochs):
    losses = []
    for epoch in range(epochs):
        predictions = activation_func(w * altitude + b)
        error = predictions - temperature

        dw = np.mean(error * altitude)
        db = np.mean(error)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        loss = np.mean(error**2)  # Mean Squared Error
        losses.append(loss)

        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {loss}")

    print(f"Final values -> w: {w:.4f}, b: {b:.4f}")
    return w, b, losses

# Step 5: Training with different activation functions
w_sigmoid, b_sigmoid, losses_sigmoid = gradient_descent(w, b, learning_rate, sigmoid, epochs)
w_relu, b_relu, losses_relu = gradient_descent(w, b, learning_rate, relu, epochs)
w_tanh, b_tanh, losses_tanh = gradient_descent(w, b, learning_rate, tanh, epochs)

# Step 6: Plot training loss for different activations
plt.figure(figsize=(10, 6))
plt.plot(losses_sigmoid, label='Sigmoid', color='blue')
plt.plot(losses_relu, label='ReLU', color='red')
plt.plot(losses_tanh, label='Tanh', color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs for Different Activation Functions')
plt.legend()
plt.grid(True)
plt.show()

# Step 7: Example prediction
new_altitude = 1500
new_altitude_normalized = (new_altitude - np.mean(altitude)) / np.std(altitude)

temp_sigmoid = sigmoid(w_sigmoid * new_altitude_normalized + b_sigmoid)
temp_relu = relu(w_relu * new_altitude_normalized + b_relu)
temp_tanh = tanh(w_tanh * new_altitude_normalized + b_tanh)

# Denormalize the predictions for display
temp_sigmoid = (temp_sigmoid * np.std(temperature)) + np.mean(temperature)
temp_relu = (temp_relu * np.std(temperature)) + np.mean(temperature)
temp_tanh = (temp_tanh * np.std(temperature)) + np.mean(temperature)

print(f"Predicted temperature at {new_altitude}m (Sigmoid): {temp_sigmoid:.2f}°C")
print(f"Predicted temperature at {new_altitude}m (ReLU): {temp_relu:.2f}°C")
print(f"Predicted temperature at {new_altitude}m (Tanh): {temp_tanh:.2f}°C")
