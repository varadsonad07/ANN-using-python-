import numpy as np

# Sample email
email = "There is big discount"

# Count capital letters
capital_letter_count = sum(1 for char in email if char.isupper())

# Check for spam words
spam_words = ["buy", "cheap", "discount", "offer"]
spam_word_count = sum(1 for word in spam_words if word in email.lower())

# Calculate email length
email_length = len(email) if len(email) < 50 else 0

# Create feature array (reshaped for training)
x = np.array([[capital_letter_count, email_length, spam_word_count]])

# Create target output array
y = np.array([1 if spam_word_count > 0 or capital_letter_count > 0 or email_length > 0 else 0])

# Initialize weights and bias
w = np.random.rand(3)
b = np.random.rand(1)

# Set learning rate and iterations
learning_rate = 0.01
iterations = 1000

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Training loop (Logistic Regression)
for i in range(iterations):
    # Compute predictions using sigmoid function
    z = np.dot(x, w) + b
    y_pred = sigmoid(z)

    # Compute error
    error = y_pred - y

    # Compute gradients
    dw = (1 / len(y)) * np.dot(x.T, error)
    db = (1 / len(y)) * np.sum(error)

    # Update weights and bias
    w -= learning_rate * dw
    b -= learning_rate * db

# Final output
print(f"Final weights: {w}")
print(f"Final bias: {b}")
print(f"Final predicted value: {y_pred}")
print(f"Final output (rounded prediction): {np.round(y_pred)}")
