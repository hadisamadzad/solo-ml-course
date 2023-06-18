from numpy import mod
import _start as starter

starter.clear_console()

import pandas as pd

# Loading
df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')

# Wrangling
df['Sex'] = df['Sex'] == 'male'

X = df[[
    'Pclass', 'Sex', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare'
]].values
y = df['Survived'].values

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# Modeling
model.fit(X, y)
print(model.coef_, model.intercept_)

# Predicting
print(model.predict([
    [3, True, 22.0, 1, 0, 7.25], \
    [3, False, 22.0, 1, 0, 7.25]]))

y_pred = model.predict(X)

# Score (Accuracy)
score = (y == y_pred).sum() / y.shape[0]
print(score)

print(model.score(X, y))
