from numpy import mod
import _start as starter

starter.clear_console()

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Loading
X, y = make_classification(n_features=2,
                           n_informative=2,
                           n_redundant=0,
                           random_state=3)

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)

# Modeling
model = MLPClassifier(
    max_iter=1000,
    hidden_layer_sizes=(100, 50),
    solver='adam',  # 'adam', 'lbfgs', 'sgd'
    activation='relu',
    alpha=0.0001,
    random_state=3)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))