# SIMPLE IMLEMENTATION OF GRADIENT DESCENT

# using numpy library to create a array list
import numpy as np
import matplotlib.pyplot as plt
x = np.array([1,2,3,4,5]) # x as input
y = np.array([2,4,6,8,10]) # y as output

w = 0 # weights
b = 0 # bias

learning_rate = 0.01 
iterations = 1000

for i in range(iterations):
    y_pred = w * x + b # predicted output
    error = y_pred - y # error

    dw = (2/len(x)) * np.sum(error * x) # gradient with respect to weights
    db = (2/len(x)) * np.sum(error) # gradient with respect to bias

    w = w - learning_rate * dw # update weight 
    b = b - learning_rate * db # update bias

    if i % 100 == 0:
        print(f"iterations is = {i} , w = {w:.4f} , b = {b:.4f} , MSE = {np.mean(error ** 2):.4f}") 

plt.scatter(x , y , color = 'blue' , label = 'True values')
plt.plot (x , w * x + b , color = 'red' , label = 'True lines')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear rigression with gradient descent')
plt.legend()
plt.show()
print(f"Final weight : {w:.4f} , Final bias {w:.4f}: ")
