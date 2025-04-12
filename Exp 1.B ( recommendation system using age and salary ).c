# simple recomendation system

def step_activation(weighted_sum , threshold):
    return "yes" if weighted_sum >= threshold else "no"

def recomendation_system(user , weights , bias):
    weighted_sum = sum(user[i] * weights[i] for i in range (len(users))) + bias
    return weighted_sum

users = [
    [25, 50000],  # User 1
    [45, 120000],  # User 2
    [18, 15000],   # User 3
]

weights = [0.1 , 0.0001]
bias = 5
threshold = 10

for user in users:
    weighted_sum = recommendation_system(user, weights, bias)
    print(f"user data : {users} -> weighted sum : {weighted_sum:.2f} -> recomendation : {step_activation(weighted_sum , threshold)}")
