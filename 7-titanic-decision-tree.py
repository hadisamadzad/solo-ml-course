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

#feature_names = ['Age', 'Siblings/Spouses', 'Fare']

X = df[feature_names].values

y = df['Survived'].values

# Train Test Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)

# Modelling
model = DecisionTreeClassifier(max_depth=3,
                               min_samples_leaf=2,
                               max_leaf_nodes=10)
model.fit(X_train, y_train)

# Prediction
#print(model.predict([[3, True, 22, 1, 0, 7.25]]))
y_pred = model.predict(X_test)
print(y_pred)

# Scores
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print(f"accuracy: {accuracy_score(y_test, y_pred)}")
print(f"precision: {precision_score(y_test, y_pred)}")
print(f"recall: {recall_score(y_test, y_pred)}")
print(f"f1: {f1_score(y_test, y_pred)}")

# Visualizing
from sklearn.tree import export_graphviz

dot_file = export_graphviz(model, feature_names=feature_names)

import graphviz

graph = graphviz.Source(dot_file)

graph.render(filename='tree', format='png', cleanup=True)
