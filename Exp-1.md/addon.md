code:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
print("Predictions:", clf.predict(X))
for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(X[i][0], X[i][1], color='red', label='Class 0' if i == 0 else "")
    else:
        plt.scatter(X[i][0], X[i][1], color='blue', label='Class 1' if i == 1 else "")
if clf.coef_[0][1] != 0:
    x_values = np.array([0, 1])
    y_values = -(clf.coef_[0][0] * x_values + clf.intercept_) / clf.coef_[0][1]
    plt.plot(x_values, y_values, label='Decision Boundary')
else:
    print("Cannot plot decision boundary: vertical line")
plt.title('Perceptron Decision Boundary for XOR')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()

output:
![deeplearning addon exp 1](https://github.com/user-attachments/assets/da5640b5-af01-4fc7-89cc-e548fa0ab61b)
