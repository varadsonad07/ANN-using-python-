#Implementation of Activation Functions: Sigmoid,ReLU, and Tanh in Neural Networks
# House Price Prediction
import numpy as np
import matplotlib.pyplot as plt

# Defining activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Creating dataset
np.random.seed(0)  # for reproducibility
house_size = np.random.uniform(500, 3000, 100)  # House size in sq ft
house_price = 100 * house_size + np.random.normal(0, 50000, 100) # House price

# Normalize data
house_size = (house_size - np.mean(house_size)) / np.std(house_size)
house_price = (house_price - np.mean(house_price)) / np.std(house_price)

# Initializing parameters
w = np.random.randn()
b = np.random.randn()
learning_rate = 0.01
epochs = 1000


# Gradient descent function
def gradient_descent(w, b, learning_rate, activation_func, epochs):
    losses = []
    for epoch in range(epochs):
        predictions = activation_func(w * house_size + b)
        error = predictions - house_price

        dw = np.mean(error * house_size)
        db = np.mean(error)

        w = w - learning_rate * dw
        b = b - learning_rate * db

        loss = np.mean(error**2)  # Mean Squared Error
        losses.append(loss)

        if epoch % 200 == 0:
            print(f"Epoch {epoch}: Loss = {loss}")

    print(f"Final values -> w: {w:.4f}, b: {b:.4f}")
    return w, b, losses


# Training with different activation functions
w_sigmoid, b_sigmoid, losses_sigmoid = gradient_descent(w, b, learning_rate, sigmoid, epochs)
w_relu, b_relu, losses_relu = gradient_descent(w, b, learning_rate, relu, epochs)
w_tanh, b_tanh, losses_tanh = gradient_descent(w, b, learning_rate, tanh, epochs)


# Plot training loss for different activations
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

# Example prediction
new_house_size = 1500
new_house_size_normalized = (new_house_size - np.mean(house_size)) / np.std(house_size)

price_sigmoid = sigmoid(w_sigmoid * new_house_size_normalized + b_sigmoid)
price_relu = relu(w_relu * new_house_size_normalized + b_relu)
price_tanh = tanh(w_tanh * new_house_size_normalized + b_tanh)

# Denormalize the predictions for display
price_sigmoid = (price_sigmoid * np.std(house_price)) + np.mean(house_price)
price_relu = (price_relu * np.std(house_price)) + np.mean(house_price)
price_tanh = (price_tanh * np.std(house_price)) + np.mean(house_price)


print(f"Predicted price for {new_house_size} sq ft (Sigmoid): {price_sigmoid:.2f}")
print(f"Predicted price for {new_house_size} sq ft (ReLU): {price_relu:.2f}")
print(f"Predicted price for {new_house_size} sq ft (Tanh): {price_tanh:.2f}")
