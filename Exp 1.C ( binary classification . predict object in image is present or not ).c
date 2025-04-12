#predict the object is present in image or not
#binary classification

def step_activation(weighted_sum , threshold):
    return 1 if weighted_sum >= threshold else 0 # 1 for object is presnet and 0 for not object

def perceptron(image , weights , bias):
    weighted_sum = sum(image[i] * weights[i] for i in range(len(image))) + bias
    return weighted_sum

images = [
    [0.8 , 0.6],
    [0.1 , 0.2],
    [0.9 , 0.7]
]
weights = [0.9 , 0.7]
bias = 0.5
threshold = 0.8

for image in images:
    weighted_sum = perceptron(image, weights, bias)
    prediction = step_activation(weighted_sum, threshold)
    print(f"Image Features: {image} -> Weighted Sum: {weighted_sum:.2f} -> Prediction: {step_activation(weighted_sum , threshold)}")
