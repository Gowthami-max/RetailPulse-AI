import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# -----------------------------------------
# PATHS
# -----------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed",
    "customer_analytics_rfm.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "models", "outputs", "churn"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------
# LOAD DATA
# -----------------------------------------

print("\n====================================")
print(" RETAILPULSE CUSTOMER CHURN MODEL")
print("====================================")

df = pd.read_csv(DATA_PATH)

print("\nOriginal dataset:", df.shape)

# -----------------------------------------
# CREATE CHURN LABEL
# -----------------------------------------

# Exclude customers who have never purchased.
df = df[
    (df["order_count"] > 0) &
    (df["days_since_last_purchase"].notna())
].copy()

# Business-defined inactivity proxy:
# 180 or more days without a purchase = inactive.

df["churn"] = (
    df["days_since_last_purchase"] >= 180
).astype(int)

print("\nChurn distribution:")
print(df["churn"].value_counts())

print("\nChurn percentage:")
print(
    (df["churn"].value_counts(normalize=True) * 100)
    .round(2)
)

# -----------------------------------------
# SELECT FEATURES
# -----------------------------------------

# Do not include recency or existing churn-related
# fields as model inputs because they define the label.

numeric_features = [
    "age",
    "order_count",
    "total_spend",
    "avg_order_value"
]

categorical_features = [
    "gender",
    "country",
    "loyalty_tier"
]

features = numeric_features + categorical_features

X = df[features].copy()
y = df["churn"]

# -----------------------------------------
# PREPROCESSING
# -----------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# -----------------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining customers:", len(X_train))
print("Testing customers:", len(X_test))

# -----------------------------------------
# RANDOM FOREST CLASSIFIER
# -----------------------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

# -----------------------------------------
# PREDICTIONS
# -----------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]

# -----------------------------------------
# EVALUATION
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

if len(np.unique(y_test)) == 2:
    auc = roc_auc_score(y_test, y_probability)
else:
    auc = None

print("\n====================================")
print(" CHURN MODEL PERFORMANCE")
print("====================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Active", "Inactive"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# -----------------------------------------
# SAVE MODEL
# -----------------------------------------

model_path = os.path.join(
    OUTPUT_DIR, "churn_prediction_model.joblib"
)

joblib.dump(model, model_path)

# -----------------------------------------
# SAVE CUSTOMER PREDICTIONS
# -----------------------------------------

customer_results = df.loc[X_test.index, [
    "customer_id",
    "customer_name",
    "order_count",
    "total_spend",
    "days_since_last_purchase"
]].copy()

customer_results["actual_churn"] = y_test
customer_results["predicted_churn"] = y_pred
customer_results["churn_probability"] = y_probability

customer_results["risk_category"] = pd.cut(
    y_probability,
    bins=[-0.001, 0.30, 0.60, 1.0],
    labels=["Low", "Medium", "High"]
)

customer_results.to_csv(
    os.path.join(OUTPUT_DIR, "churn_predictions.csv"),
    index=False
)

# -----------------------------------------
# SAVE METRICS
# -----------------------------------------

metrics = {
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(auc) if auc is not None else None,
    "training_customers": int(len(X_train)),
    "testing_customers": int(len(X_test)),
    "inactivity_threshold_days": 180,
    "target_definition": "180 or more days since last purchase"
}

with open(
    os.path.join(OUTPUT_DIR, "churn_metrics.json"),
    "w"
) as file:
    json.dump(metrics, file, indent=4)

print("\n====================================")
print(" CHURN MODEL SAVED SUCCESSFULLY")
print("====================================")

print("Model:", model_path)
print("Predictions: churn_predictions.csv")
print("Metrics: churn_metrics.json")