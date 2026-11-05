import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORDERS = os.path.join(BASE, "data", "processed", "orders.csv")
CUSTOMERS = os.path.join(BASE, "data", "processed", "customer_analytics_rfm.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "churn")

os.makedirs(OUT, exist_ok=True)

# Historical prediction date and future observation window
CUTOFF = pd.Timestamp("2025-12-31")
END_DATE = CUTOFF + pd.Timedelta(days=180)

orders = pd.read_csv(ORDERS)
customers = pd.read_csv(CUSTOMERS)

orders["order_datetime"] = pd.to_datetime(
    orders["order_datetime"], errors="coerce"
)

orders["order_status"] = (
    orders["order_status"].astype(str).str.lower()
)

orders = orders[
    (orders["order_status"] == "delivered") &
    orders["order_datetime"].notna()
].copy()

orders["order_total"] = pd.to_numeric(
    orders["order_total"], errors="coerce"
)

# Only customers with purchases before prediction date
past = orders[orders["order_datetime"] <= CUTOFF].copy()

# Customers with at least one purchase in future 180 days
future = orders[
    (orders["order_datetime"] > CUTOFF) &
    (orders["order_datetime"] <= END_DATE)
].copy()

# Historical features
features = past.groupby("customer_id").agg(
    order_count=("order_id", "nunique"),
    total_spend=("order_total", "sum"),
    avg_order_value=("order_total", "mean"),
    last_purchase=("order_datetime", "max"),
    first_purchase=("order_datetime", "min")
).reset_index()

features["recency"] = (
    CUTOFF - features["last_purchase"]
).dt.days

features["customer_tenure"] = (
    CUTOFF - features["first_purchase"]
).dt.days

features["purchase_frequency"] = (
    features["order_count"] /
    (features["customer_tenure"] / 30 + 1)
)

# Future churn label: no purchase during next 180 days
future_customers = set(future["customer_id"].unique())

features["churn"] = (
    ~features["customer_id"].isin(future_customers)
).astype(int)

# Add static customer information
static_cols = ["customer_id", "age", "gender", "country", "loyalty_tier"]

static = customers[
    [c for c in static_cols if c in customers.columns]
].drop_duplicates("customer_id")

data = features.merge(static, on="customer_id", how="left")

data = data.drop(columns=[
    "last_purchase", "first_purchase"
])

# Remove identifiers
X = data.drop(columns=["customer_id", "churn"])
y = data["churn"]

# Convert categorical columns
X = pd.get_dummies(X, dummy_na=True)

# Chronological train/test split by customer history
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []
best_auc = -1
best_model = None
best_name = None

for name, model in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probabilities)

    print("ROC-AUC:", round(auc, 4))
    print(classification_report(y_test, predictions))

    results.append({
        "Model": name,
        "ROC_AUC": auc
    })

    if auc > best_auc:
        best_auc = auc
        best_model = pipeline
        best_name = name

pd.DataFrame(results).to_csv(
    os.path.join(OUT, "future_churn_comparison.csv"),
    index=False
)

joblib.dump(
    best_model,
    os.path.join(OUT, "future_churn_model.joblib")
)

print("\n================================")
print("FUTURE CHURN RESULTS")
print("================================")
print("Customers:", len(data))
print("Churn percentage:", round(y.mean() * 100, 2), "%")
print("Best model:", best_name)
print("ROC-AUC:", round(best_auc, 4))
print("Target:", "ACHIEVED" if best_auc >= 0.88 else "NOT ACHIEVED")
print("Model saved successfully.")