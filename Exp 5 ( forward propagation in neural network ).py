!pip install openpyxl

import openpyxl

from google.colab import files
uploaded = files.upload()

from google.colab import drive
drive.mount('/content/drive')

!ls /content/drive/MyDrive/ 

!ls /content/drive/MyDrive/Classroom

!ls /content/drive/MyDrive/Classroom/SY-2024-25-AIML-Div\ B-ANN

!ls /content/drive/MyDrive/Classroom/SY-2024-25-AIML-Div\ B-ANN/ Student_dataset.xlsx

import pandas as pd
df = pd.read_excel('/content/drive/MyDrive/Classroom/SY-2024-25-AIML-Div B-ANN/Student_dataset.xlsx') 
df.head()

import numpy as np

# Initialize parameters
def initialize_parameters(layer_dims):
    np.random.seed(3)
    print("Layer Dimensions:", layer_dims)
    parameters = {}
    L = len(layer_dims)
    print("Total No. of layers in NN", L)
    for i in range(1, L):
        parameters['W' + str(i)] = np.ones((layer_dims[i-1], layer_dims[i])) * 0.1
        parameters['b' + str(i)] = np.zeros((layer_dims[i], 1))
    return parameters


# Forward propagation
def linear_forward(A_prev, W, b):
    Z = np.dot(W.T, A_prev) + b
    return Z

def relu(Z):
    return np.maximum(0, Z)

def L_layer_forward(X, parameters):
    A = X
    caches = []
    print("\n", parameters)
    L = len(parameters) // 2
    for i in range(1, L):
        A_prev = A 
        W = parameters['W' + str(i)]
        b = parameters['b' + str(i)]
        Z = linear_forward(A_prev, W, b)
        A = relu(Z)
        cache = (A_prev, W, b, Z)
        caches.append(cache)

 # Output layer
    W_out = parameters['W' + str(L)]
    b_out = parameters['b' + str(L)]
    Z_out = linear_forward(A, W_out, b_out)
    AL = Z_out
    return AL, caches

# Example execution
layer_dims = [2, 2, 1]  # 2 inputs, 2 hidden neurons, 1 output neuron
parameters = initialize_parameters(layer_dims)

# Example input (replace df with a valid DataFrame or just use a numpy array directly)
X = np.array([[7.0], [3.5]])  # Sample input (e.g., CGPA and profile_score)

y_hat, caches = L_layer_forward(X, parameters)
print("Final output:")
print(y_hat)
