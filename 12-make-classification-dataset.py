import _start as starter

starter.clear_console()

# Loading
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=100,
                           n_features=20,
                           n_redundant=0,
                           n_informative=2,
                           random_state=3)

# Visualizing
from matplotlib import pyplot as plt

plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], s=100, edgecolors='k')
plt.scatter(X[y == 1][:, 0],
            X[y == 1][:, 1],
            s=100,
            edgecolors='k',
            marker='^')
plt.savefig('classification_dataset.png')