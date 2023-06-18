import _start as starter

starter.clear_console()

from pandas.core.frame import DataFrame
from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()

df = DataFrame(cancer_data.data, columns=cancer_data.feature_names)

df['target'] = cancer_data.target

X = df[cancer_data.feature_names].values
y = df['target'].values

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear')

model.fit(X, y)

print(model.score(X, y))
