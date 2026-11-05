import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "customer_analytics_rfm.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "models", "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load dataset
print("Loading customer dataset...")

df = pd.read_csv(DATA_PATH)

print("Total customers:", len(df))


# Select numerical features
FEATURES = [
    "order_count",
    "total_spend",
    "avg_order_value",
    "days_since_last_purchase"
]


# Separate customers with no purchases
no_purchase = df[df["order_count"] == 0].copy()

customers = df[df["order_count"] > 0].copy()

print("Customers with purchases:", len(customers))
print("Customers without purchases:", len(no_purchase))


# Handle missing values
customers[FEATURES] = customers[FEATURES].fillna(0)

# Limit extreme recency values
recency_cap = customers["days_since_last_purchase"].quantile(0.99)

customers["days_since_last_purchase"] = (
    customers["days_since_last_purchase"].clip(upper=recency_cap)
)


# Log transformation reduces skewness
X = customers[FEATURES].copy()

X = np.log1p(X)


# Feature scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Find the best number of clusters
print("\nEvaluating K-Means clusters...")

results = []

for k in range(2, 7):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    results.append({
        "clusters": k,
        "silhouette_score": score
    })

    print(f"K={k} | Silhouette Score={score:.4f}")


# Select the best clustering model
best_result = max(
    results,
    key=lambda x: x["silhouette_score"]
)

best_k = best_result["clusters"]

print("\nBest number of clusters:", best_k)


# Train final model
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

customers["cluster"] = kmeans.fit_predict(X_scaled)


# Generate descriptive cluster names
cluster_stats = customers.groupby("cluster").agg(
    customers=("customer_id", "count"),
    avg_spending=("total_spend", "mean"),
    avg_orders=("order_count", "mean"),
    avg_order_value=("avg_order_value", "mean")
)

ranked_clusters = cluster_stats.sort_values(
    "avg_spending"
).index.tolist()

segment_names = {}

names = [
    "Low Value",
    "Regular Customers",
    "High Value",
    "Top Value"
]

for index, cluster_id in enumerate(ranked_clusters):

    if len(ranked_clusters) == 2:
        name = ["Regular Customers", "High Value"][index]

    elif len(ranked_clusters) == 3:
        name = [
            "Low Value",
            "Regular Customers",
            "High Value"
        ][index]

    else:
        name = names[min(index, len(names) - 1)]

    segment_names[cluster_id] = name


customers["customer_segment"] = customers["cluster"].map(
    segment_names
)

# Customers without purchase history
no_purchase["cluster"] = -1

no_purchase["customer_segment"] = "No Purchase"


# Combine results
final_df = pd.concat(
    [customers, no_purchase],
    ignore_index=True
)

final_df.to_csv(
    os.path.join(OUTPUT_DIR, "customer_segmentation_results.csv"),
    index=False
)


# Cluster summary
summary = customers.groupby("customer_segment").agg(
    customer_count=("customer_id", "count"),
    average_spending=("total_spend", "mean"),
    average_orders=("order_count", "mean"),
    average_order_value=("avg_order_value", "mean")
).round(2)

summary.to_csv(
    os.path.join(OUTPUT_DIR, "customer_segment_summary.csv")
)


# Save model
joblib.dump(
    {
        "scaler": scaler,
        "kmeans": kmeans,
        "features": FEATURES,
        "recency_cap": recency_cap,
        "segment_names": segment_names
    },
    os.path.join(OUTPUT_DIR, "customer_segmentation_model.joblib")
)


# Save evaluation metrics
silhouette = best_result["silhouette_score"]

metrics = {
    "total_customers": len(df),
    "customers_with_purchases": len(customers),
    "customers_without_purchases": len(no_purchase),
    "optimal_clusters": int(best_k),
    "silhouette_score": float(silhouette),
    "cluster_evaluation": results
}

with open(
    os.path.join(OUTPUT_DIR, "segmentation_metrics.json"),
    "w"
) as file:

    json.dump(metrics, file, indent=4)


# Final output
print("\n================================")
print("CUSTOMER SEGMENTATION COMPLETED")
print("================================")

print("\nCluster Summary:")
print(summary)

print("\nSilhouette Score:", round(silhouette, 4))

print("\nGenerated files:")
print("1. customer_segmentation_results.csv")
print("2. customer_segment_summary.csv")
print("3. customer_segmentation_model.joblib")
print("4. segmentation_metrics.json")

print("\nModel training completed successfully!")