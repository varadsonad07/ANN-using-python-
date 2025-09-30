# From Altitude Tempreture prediction:-
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset (Altitude vs. Temprature)
# Example dataset: Study hours (x) and corresponding exam scores (y)
x = np.array([0,20,40,60,80,100,120])  # Altitude
y = np.array([0,5,10,15,20,25,30])  # Tempreture

# Initialize parameters
m = 0  # Slope
b = 0 # Intercept
learning_rate = 0.0001
iterations = 100 # Number of iterations

# Number of data points
n = len(x)

# Gradient Descent implementation
for iterations in range(iterations):
    # Predictions
  y_pred = m * x + b

    # Calculate gradients
  cost=(1/n) * sum((y-y_pred) * 2 )
  dm = -(2 / n) * sum(x * (y - y_pred))  # Partial derivative with respect to m
  db = -(2 / n) * sum(y - y_pred)       # Partial derivative with respect to b

    # Update parameters
  m -= learning_rate * dm
  b -= learning_rate * db
  print("\nm=",m)
  print("\tb=",b)
  print("\tcost=",cost)
# Predict exam scores based on study hours
def predict(temp):
    return m * temp + b

# Visualize the dataset and the best-fit line
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, predict(x), color='red', label='Best-fit Line')
plt.xlabel('Altitude')
plt.ylabel('Tempreture')
plt.title('Altitude Tempreture predictions')
plt.legend()
plt.show()

# Example prediction
altitude = 110
predicted_temp = predict(altitude)
print("\nfinal output : ",predicted_temp)
