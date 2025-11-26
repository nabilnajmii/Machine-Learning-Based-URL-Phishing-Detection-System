# src/train.py
# ------------------------------------------------------------
# Train baseline models on URL lexical features and save best model.
# ------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from joblib import dump

# allow "from features import ..." when run from project root
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from features import extract_features, feature_order  # noqa


DATA_PHISH = os.path.join("data", "phishing.csv")
DATA_LEGIT = os.path.join("data", "legit.csv")
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data():
    """Load CSVs, add labels: 1=phish, 0=legit; merge & dedupe."""
    phish = pd.read_csv(DATA_PHISH).dropna(subset=["url"]).copy()
    legit = pd.read_csv(DATA_LEGIT).dropna(subset=["url"]).copy()
    phish["label"] = 1
    legit["label"] = 0
    df = pd.concat([phish, legit], ignore_index=True).drop_duplicates("url")
    return df


def build_xy(df: pd.DataFrame):
    """Extract features for each URL and build X (numpy) and y labels."""
    feats = df["url"].apply(extract_features).tolist()
    order = feature_order()
    X = np.array([[d[k] for k in order] for d in feats], dtype=float)
    y = df["label"].to_numpy(dtype=int)
    return X, y


def candidate_models():
    """Define 3 baseline models using scikit-learn Pipelines."""
    return {
        "DecisionTree": Pipeline([("clf", DecisionTreeClassifier(random_state=42))]),
        "RandomForest": Pipeline([("clf", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42))]),
        "SVM_RBF": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True, random_state=42))]),
    }


def evaluate(model, X_test, y_test, name):
    """Train finished; compute metrics on test set and print them."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print(f"\n{name} results:")
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall   : {rec:.3f}")
    print(f"  F1-score : {f1:.3f}")
    return f1


def main():
    df = load_data()
    X, y = build_xy(df)

    # small dataset => bigger test split to see variability
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, #20% testing, 80% training
        stratify=y, #ensure balance of phishing and legit urls for both training and testing sets
        random_state=42 
    )

    print(f"\n[Info] Dataset split:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples : {len(X_test)}")
    print(f"  Total samples   : {len(X)}")


    best_name, best_model, best_f1 = None, None, -1.0
    for name, pipe in candidate_models().items():
        pipe.fit(X_train, y_train)
        f1 = evaluate(pipe, X_test, y_test, name)
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, pipe, f1

    # Save best model + feature order
    dump(best_model, os.path.join(MODELS_DIR, "best_model.joblib"))
    pd.Series(feature_order()).to_csv(os.path.join(MODELS_DIR, "feature_order.csv"), index=False)
    print(f"\n✅ Saved best model: {best_name} (F1={best_f1:.3f}) → models/best_model.joblib")


if __name__ == "__main__":
    main()
