import _start as starter

starter.clear_console()

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Loading
import numpy as np
from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# Splitting
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)

# Modeling
model = MLPClassifier(
    max_iter=200,
    hidden_layer_sizes=(6, ),
    solver='sgd',  # 'adam', 'lbfgs', 'sgd'
    #activation='relu',
    alpha=1e-4,
    random_state=2)

model.fit(X, y)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(5, 4))
for i, ax in enumerate(axes.ravel()):
    coef = model.coefs_[0][:, i]
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(i + 1)
plt.savefig('ann-hidden-layer.png')