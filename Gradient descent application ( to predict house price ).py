# From Square feet predict House values:-
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset (Square values vs. House values)
# Example dataset: Square feet values(x) and corresponding House values (y)
x = np.array([800,850,900,950,1000])  # Square feet values
y = np.array([16000,17000,18000,19000,20000])  # House values

# Initialize parameters
m = 0  # Slope
b = 0 # Intercept
learning_rate = 0.000001
iterations = 110 # Number of iterations

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
def predict(house):
    return m * house + b

# Visualize the dataset and the best-fit line
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, predict(x), color='red', label='Best-fit Line')
plt.xlabel('Square Feet Values')
plt.ylabel('House values')
plt.title('House Value predictions')
plt.legend()
plt.show()

# Example prediction
Square = 920
predicted_house = predict(Square)
print("\nfinal output : ",predicted_house)
