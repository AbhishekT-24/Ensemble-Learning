from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTree import DecisionTreeClassifier
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from DecisionTree import DecisionTreeRegressor
from sklearn.metrics import r2_score


data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

clf = DecisionTreeClassifier(max_depth=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(acc)

#for regression task
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree regressor on the training set
dt = DecisionTreeRegressor(max_depth=3, min_samples_split=12)
dt.fit(X_train, y_train)

# Evaluate the model on the testing set
y_pred = dt.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)

r2 = r2_score(y_test, y_pred)
print("R2 score:", r2)

#Random Forrest Implementation for Regression
n_trees = 100
forest = []
for i in range(n_trees):
    # Subsample the training data
    idx = np.random.choice(X_train.shape[0], size=X_train.shape[0], replace=True)
    X_train_sub = X_train[idx,:]
    y_train_sub = y_train[idx]
    # Create a decision tree regressor and fit it to the subsampled data
    dt = DecisionTreeRegressor(max_depth=3, min_samples_split=6)
    dt.fit(X_train_sub, y_train_sub)
    # Add the decision tree to the forest
    forest.append(dt)

# Evaluate the random forest on the testing set
y_pred = np.zeros(X_test.shape[0])
for dt in forest:
    y_pred += dt.predict(X_test)
y_pred /= n_trees
r2 = r2_score(y_test, y_pred)
print("R2 score: Random Forrest", r2)







