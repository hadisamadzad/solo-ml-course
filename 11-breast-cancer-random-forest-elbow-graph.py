import _start as starter

starter.clear_console()

from pandas.core.frame import DataFrame

# Loading
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()

# Wrangling
df = DataFrame(cancer_data.data, columns=cancer_data.feature_names)
df['target'] = cancer_data.target

X = df[cancer_data.feature_names].values
y = df['target'].values

# Train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101)

# Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

n_estimators = list(range(1, 101))
param_grid = {
    'n_estimators': n_estimators,
}
rf = RandomForestClassifier()
gs = GridSearchCV(rf, param_grid, cv=5)
gs.fit(X, y)

scores = gs.cv_results_['mean_test_score']

print("score:", scores)

# Visualization
import matplotlib.pyplot as plt

plt.plot(n_estimators, scores)
plt.xlabel("n_estimators")
plt.ylabel("accuracy")
plt.xlim(0, 40)
plt.ylim(0.9, 1)
plt.show()

plt.savefig('elbow.png')