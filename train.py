import re
import mlflow, mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def dvc_hash(path):
    text = open(path + ".dvc").read()
    return re.search(r"md5:\s*(\S+)", text).group(1)

df = pd.read_csv("data/train.csv")
X, y = df.drop(columns=["target"]), df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

mlflow.set_experiment("DVC_MLflow_Demo")
with mlflow.start_run():
    mlflow.log_param("dataset_version", dvc_hash("data/train.csv"))
    mlflow.log_param("rows", len(df))
    mlflow.log_param("n_estimators", 100)
    model = RandomForestClassifier(n_estimators=100, max_depth=3)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "random_forest_model")
    print(f"Accuracy: {acc:.4f}")
