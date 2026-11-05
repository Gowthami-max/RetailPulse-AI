import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "customer_analytics_rfm.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "models",
    "outputs",
    "final_segmentation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_CLUSTERS = 6

FEATURES = [
    "days_since_last_purchase",
    "order_count",
    "total_spend"
]

# ==================================================
# 2. BUSINESS SEGMENT DEFINITIONS
# ==================================================

SEGMENT_NAMES = {
    0: "Hibernating Customers",
    1: "At-Risk Customers",
    2: "Recent Customers",
    3: "High-Value Loyalists",
    4: "Champions",
    5: "Potential Loyalists"
}

# ==================================================
# 3. LOAD CUSTOMER DATA
# ==================================================

print("\nLoading customer dataset...")

df = pd.read_csv(DATA_PATH)

print("Total customers:", len(df))

# Separate customers with purchase history

purchasing = df[df["order_count"] > 0].copy()

no_purchase = df[df["order_count"] == 0].copy()

print("Customers with purchases:", len(purchasing))
print("Customers without purchases:", len(no_purchase))

# ==================================================
# 4. DATA PREPROCESSING
# ==================================================

purchasing[FEATURES] = purchasing[FEATURES].apply(
    pd.to_numeric,
    errors="coerce"
)

purchasing[FEATURES] = purchasing[FEATURES].fillna(0)

# Recency should not be negative
purchasing["days_since_last_purchase"] = (
    purchasing["days_since_last_purchase"].clip(lower=0)
)

# Log transformation
X = np.log1p(purchasing[FEATURES])

# Standard scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==================================================
# 5. TRAIN FINAL K-MEANS MODEL
# ==================================================

print("\nTraining six-cluster K-Means model...")

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=20
)

purchasing["cluster"] = kmeans.fit_predict(X_scaled)

# ==================================================
# 6. GENERATE BUSINESS SEGMENT NAMES
# ==================================================

purchasing["customer_segment"] = (
    purchasing["cluster"].map(SEGMENT_NAMES)
)

# Customers without purchase history
no_purchase["cluster"] = -1

no_purchase["customer_segment"] = "No Purchase"

# ==================================================
# 7. GENERATE CUSTOMER-LEVEL OUTPUT
# ==================================================

final_df = pd.concat(
    [purchasing, no_purchase],
    ignore_index=True
)

final_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_customer_segments.csv"
    ),
    index=False
)

# ==================================================
# 8. GENERATE SEGMENT PROFILES
# ==================================================

summary = purchasing.groupby(
    ["cluster", "customer_segment"]
).agg(
    customer_count=("customer_id", "count"),
    average_recency=("days_since_last_purchase", "mean"),
    median_recency=("days_since_last_purchase", "median"),
    average_frequency=("order_count", "mean"),
    median_frequency=("order_count", "median"),
    average_spending=("total_spend", "mean"),
    median_spending=("total_spend", "median"),
    average_order_value=("avg_order_value", "mean")
).round(2)

summary["customer_percentage"] = (
    summary["customer_count"] /
    len(purchasing) * 100
).round(2)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_segment_profiles.csv"
    )
)

# ==================================================
# 9. MODEL EVALUATION
# ==================================================

print("\nEvaluating final segmentation...")

silhouette = silhouette_score(
    X_scaled,
    purchasing["cluster"],
    sample_size=3000,
    random_state=RANDOM_STATE
)

davies_bouldin = davies_bouldin_score(
    X_scaled,
    purchasing["cluster"]
)

calinski_harabasz = calinski_harabasz_score(
    X_scaled,
    purchasing["cluster"]
)

cluster_sizes = (
    purchasing["cluster"]
    .value_counts()
    .sort_index()
)

# ==================================================
# 10. SAVE MODEL
# ==================================================

model_artifact = {
    "model": kmeans,
    "scaler": scaler,
    "features": FEATURES,
    "n_clusters": N_CLUSTERS,
    "segment_names": SEGMENT_NAMES,
    "random_state": RANDOM_STATE
}

joblib.dump(
    model_artifact,
    os.path.join(
        OUTPUT_DIR,
        "final_customer_segmentation_model.joblib"
    )
)

# ==================================================
# 11. SAVE METRICS
# ==================================================

metrics = {
    "total_customers": int(len(df)),
    "customers_with_purchases": int(len(purchasing)),
    "customers_without_purchases": int(len(no_purchase)),
    "algorithm": "K-Means",
    "number_of_clusters": N_CLUSTERS,
    "features": FEATURES,
    "silhouette_score": float(silhouette),
    "davies_bouldin_score": float(davies_bouldin),
    "calinski_harabasz_score": float(calinski_harabasz),
    "cluster_sizes": {
        str(k): int(v)
        for k, v in cluster_sizes.items()
    },
    "random_state": RANDOM_STATE
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "final_segmentation_metrics.json"
    ),
    "w"
) as file:

    json.dump(metrics, file, indent=4)

# ==================================================
# 12. FINAL REPORT
# ==================================================

print("\n==========================================")
print("FINAL CUSTOMER SEGMENTATION COMPLETED")
print("==========================================")

print("\nEvaluation Metrics:")

print("Silhouette Score:", round(silhouette, 4))
print("Davies-Bouldin Score:", round(davies_bouldin, 4))
print("Calinski-Harabasz Score:", round(calinski_harabasz, 2))

print("\nCustomer Segment Summary:")
print(summary.to_string())

print("\nGenerated Files:")
print("1. final_customer_segments.csv")
print("2. final_segment_profiles.csv")
print("3. final_customer_segmentation_model.joblib")
print("4. final_segmentation_metrics.json")

print("\nOutput Directory:", OUTPUT_DIR)

print("\nModel training completed successfully!")