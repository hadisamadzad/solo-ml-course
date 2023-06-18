import _start as starter

starter.clear_console()

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'

feature_names = [
    'Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare'
]

X = df[feature_names].values

y = df['Survived'].values

# Train Test Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

# Grid search
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [5, 15, 25],
    'min_samples_leaf': [1, 3],
    'max_leaf_nodes': [10, 20, 35, 50]
}
dt = DecisionTreeClassifier()
gs = GridSearchCV(dt, param_grid, scoring='f1', cv=5)
gs.fit(X, y)

print(gs.best_params_)
print(gs.best_score_)
