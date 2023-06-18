import _start as starter

starter.clear_console()

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from pandas import DataFrame

# Loading
X, y = load_digits(n_class=2, return_X_y=True)
print(X[0].reshape(8, 8))

import matplotlib.pyplot as plt

# Early visualization
plt.matshow(X[219].reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())  # remove x tick marks
plt.yticks(())  # remove y tick marks
plt.savefig('sample-2digit.png')

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

# Modeling
model = MLPClassifier(
    max_iter=200,
    #hidden_layer_sizes=(100, 50),
    #solver='adam',  # 'adam', 'lbfgs', 'sgd'
    #activation='relu',
    #alpha=0.0001,
    #random_state=3
)

model.fit(X_train, y_train)

# Prediction

x = X_test[1]
plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.savefig('predicted-2digit.png')
y = model.predict([x])
print(y)

print(model.score(X_test, y_test))