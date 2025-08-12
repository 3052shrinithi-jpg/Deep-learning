code:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [1], [1], [0]])
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y, epochs=1000, verbose=0)
loss, acc = model.evaluate(X, Y, verbose=0)
print("Accuracy:", acc)
predictions = model.predict(X)
print("\nPredictions (Raw):")
print(predictions)
print("\nPredictions (Rounded):")
for i, pred in enumerate(np.round(predictions)):
    print(f"Input: {X[i]} -> Output: {int(pred[0])}")
plt.plot(history.history['loss'])
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

output:

![deeplearning core exp no 1](https://github.com/user-attachments/assets/c0d3be54-3dba-46d4-a383-c5177421b886)
