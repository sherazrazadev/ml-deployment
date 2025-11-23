import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data (Source: Internal Scikit-learn dataset)
iris = load_iris()
X, y = iris.data, iris.target

# 2. Train Model
# We use RandomForest as it's robust and simple for this demo
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X, y)

# 3. Save the Model
# joblib is more efficient than pickle for numpy-heavy models
joblib.dump(clf, "app/model.joblib")
print("Model trained and saved to app/model.joblib")