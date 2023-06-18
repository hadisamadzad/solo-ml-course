import _start as starter

starter.clear_console()

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from pandas import DataFrame

# Loading
X, y = load_digits(return_X_y=True)

import matplotlib.pyplot as plt

# Early visualization
plt.matshow(X[219].reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())  # remove x tick marks
plt.yticks(())  # remove y tick marks
plt.savefig('sample-10digit.png')

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

# Modeling
model = MLPClassifier(
    max_iter=200,
    hidden_layer_sizes=(200, 50),
    #solver='adam',  # 'adam', 'lbfgs', 'sgd'
    #activation='relu',
    #alpha=0.0001,
    random_state=2)

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

incorrect = X_test[y_pred != y_test]
incorrect_true = y_test[y_pred != y_test]
incorrect_pred = y_pred[y_pred != y_test]

i = 0
plt.matshow(incorrect[i].reshape(8, 8), cmap=plt.cm.gray)
plt.xticks(())
plt.yticks(())
plt.savefig('predicted-10digit.png')

print("true value:", incorrect_true[i])
print("predicted value:", incorrect_pred[i])

print(model.score(X_test, y_test))
