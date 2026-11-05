import os
import warnings
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(
    BASE, "data", "processed", "customer_analytics_rfm.csv"
)

OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs", "churn"
)

os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(DATA)

df = df.dropna(subset=["days_since_last_purchase"]).copy()

df["churn"] = (
    df["days_since_last_purchase"] >= 180
).astype(int)

# Remove target, direct leakage and identifiers
excluded = [
    "customer_id",
    "customer_name",
    "email",
    "phone",
    "churn",
    "days_since_last_purchase",
    "inactivity_risk",
    "rfm_segment",
    "last_purchase_date",
    "first_purchase_date",
    "signup_date"
]

X = df.drop(columns=[
    c for c in excluded if c in df.columns
])

y = df["churn"]

numeric = X.select_dtypes(include="number").columns.tolist()
categorical = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )
}

results = []
best_auc = -1
best_model = None
best_name = None

for name, classifier in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1": f1_score(y_test, predictions),
        "ROC_AUC": auc
    })

    print("ROC-AUC:", round(auc, 6))

    if auc > best_auc:
        best_auc = auc
        best_model = pipeline
        best_name = name

pd.DataFrame(results).to_csv(
    os.path.join(OUT, "churn_leakage_free_comparison.csv"),
    index=False
)

joblib.dump(
    best_model,
    os.path.join(OUT, "churn_leakage_free_model.joblib")
)

print("\n========== FINAL RESULTS ==========")
print(pd.DataFrame(results).to_string(index=False))
print("\nBest model:", best_name)
print("Best ROC-AUC:", round(best_auc, 6))
print("Target:", "ACHIEVED" if best_auc >= 0.88 else "NOT ACHIEVED")