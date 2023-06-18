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

model = RandomForestClassifier()
#model = RandomForestClassifier(max_features=5, n_estimators=10)

model.fit(X_train, y_train)

# Prediction
first_row = X_test[0]
print("prediction:", model.predict([first_row]))
print("expectatioin:", y_test[0])

# Scores
print("score:", model.score(X_test, y_test))
