from pandas.core.frame import DataFrame
import _start as starter

starter.clear_console()

import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('https://sololearn.com/uploads/files/titanic.csv')
df['male'] = df['Sex'] == 'male'

X = df[[
    'Pclass', 'male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare'
]].values

y = df['Survived'].values

# Modeling
model = LogisticRegression()
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# Confusion matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y, y_pred))

# Scores
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# prediction accuracy
print(f"accuracy: {accuracy_score(y, y_pred)}")

# precision about positive preds
print(f"precision: {precision_score(y, y_pred)}")

# precision about actual positives
print(f"recall: {recall_score(y, y_pred)}")
print(f"f1: {f1_score(y, y_pred)}")
