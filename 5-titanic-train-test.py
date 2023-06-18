import _start as starter

starter.clear_console()

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'

X = df[[
    'Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare'
]].values

y = df['Survived'].values

# Train Test Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=25)

# Modelling
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Scores
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(f"accuracy: {accuracy_score(y_test, y_pred)}")
print(f"precision: {precision_score(y_test, y_pred)}")
print(f"recall: {recall_score(y_test, y_pred)}")
print(f"f1: {f1_score(y_test, y_pred)}")
