#Bank Loan Approval
import numpy as np

x1 = [720,650,780,600,680]  # Credit score range
x2 = [50000,40000,60000,30000,45000]  # Annual income range
x3 = [0, 1]  # Loan approval status (0: Not Approved, 1: Approved)

# Input from the user
xn = int(input("Enter credit score: "))
In = int(input("Enter annual income: "))

# Feature scaling
xcred = (xn - min(x1)) / (max(x1) - min(x1))
xin = (In - min(x2)) / (max(x2) - min(x2))

# Adjustable weights and threshold
credit_weight = 1  
income_weight = 0.5  
threshold = 0.7  

# Loan approval prediction
y = credit_weight * xcred + income_weight * xin  
# Removed random bias for a deterministic approach

if y > threshold:
  print("Loan is approved")
else:
  print("Loan is not approved")
