#sigmoid activation function using gradient descent

import numpy as np

def Sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def Gradient_descent(x, y, learning_rate, epochs):
    weight = 0.2
    for _ in range(epochs):
        y_pred = Sigmoid_function(x * weight)
        error = y - y_pred
        weight += learning_rate * np.sum(error * x)
    return weight

x = np.array([-2, 1, 4, 5, -6])
y = np.array([1, 0, 1, 1, 0])
learning_rate = 0.01
epochs = 100

final_weight = Gradient_descent(x, y, learning_rate, epochs)
y_pred = Sigmoid_function(x * final_weight)
print(f"Final weight: {final_weight:.2f}")
print(f"y_prediction is: {np.round(y_pred)}")
