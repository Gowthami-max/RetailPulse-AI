import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
    "segmentation_experiment"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

print("\nLoading customer dataset...")

df = pd.read_csv(DATA_PATH)

customers = df[df["order_count"] > 0].copy()

FEATURES = [
    "days_since_last_purchase",
    "order_count",
    "total_spend"
]

customers[FEATURES] = customers[FEATURES].fillna(0)

print("Total customers:", len(df))
print("Customers with purchases:", len(customers))
print("Customers without purchases:", len(df) - len(customers))

# --------------------------------------------------
# 3. TRANSFORM AND SCALE RFM FEATURES
# --------------------------------------------------

X = customers[FEATURES].copy()

X = np.log1p(X)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Use the same fixed sample for fair model comparisons
rng = np.random.RandomState(42)

sample_size = min(3000, len(X_scaled))

sample_indices = rng.choice(
    len(X_scaled),
    size=sample_size,
    replace=False
)

X_sample = X_scaled[sample_indices]

# --------------------------------------------------
# 4. EVALUATION FUNCTION
# --------------------------------------------------

def evaluate_clustering(X_data, labels, method, parameters):

    labels = np.asarray(labels)

    # Exclude DBSCAN noise points (-1) from quality metrics
    valid_mask = labels != -1

    valid_labels = labels[valid_mask]
    valid_data = X_data[valid_mask]

    unique_clusters = np.unique(valid_labels)

    cluster_count = len(unique_clusters)

    noise_count = int(np.sum(labels == -1))

    result = {
        "method": method,
        "parameters": str(parameters),
        "clusters": int(cluster_count),
        "noise_points": noise_count,
        "silhouette_score": None,
        "davies_bouldin_score": None,
        "calinski_harabasz_score": None
    }

    if cluster_count < 2:
        return result

    if cluster_count >= len(valid_labels):
        return result

    # Evaluate on a reproducible sample for efficiency
    if len(valid_labels) > 3000:

        local_rng = np.random.RandomState(42)

        indices = local_rng.choice(
            len(valid_labels),
            3000,
            replace=False
        )

        eval_data = valid_data[indices]
        eval_labels = valid_labels[indices]

        # Sample must contain at least two clusters
        if len(np.unique(eval_labels)) < 2:
            return result

    else:
        eval_data = valid_data
        eval_labels = valid_labels

    result["silhouette_score"] = float(
        silhouette_score(eval_data, eval_labels)
    )

    result["davies_bouldin_score"] = float(
        davies_bouldin_score(eval_data, eval_labels)
    )

    result["calinski_harabasz_score"] = float(
        calinski_harabasz_score(eval_data, eval_labels)
    )

    return result


results = []

# --------------------------------------------------
# 5. K-MEANS: TEST 2 TO 8 CLUSTERS
# --------------------------------------------------

print("\n========== K-MEANS EVALUATION ==========")

best_kmeans_score = -1
best_kmeans_model = None
best_kmeans_labels = None
best_k = None

for k in range(2, 9):

    print(f"\nTesting K-Means with K={k}")

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X_scaled)

    metrics = evaluate_clustering(
        X_sample,
        model.predict(X_sample),
        "KMeans",
        {"k": k}
    )

    results.append(metrics)

    score = metrics["silhouette_score"]

    print(metrics)

    if score is not None and score > best_kmeans_score:

        best_kmeans_score = score
        best_kmeans_model = model
        best_kmeans_labels = labels
        best_k = k

# --------------------------------------------------
# 6. DBSCAN: TEST MULTIPLE PARAMETERS
# --------------------------------------------------

print("\n========== DBSCAN EVALUATION ==========")

eps_values = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.0]

min_samples_values = [10, 20, 30, 50]

best_dbscan_score = -1
best_dbscan_model = None
best_dbscan_labels = None
best_dbscan_params = None

for eps in eps_values:

    for min_samples in min_samples_values:

        model = DBSCAN(
            eps=eps,
            min_samples=min_samples
        )

        labels = model.fit_predict(X_scaled)

        metrics = evaluate_clustering(
            X_sample,
            model.fit_predict(X_sample),
            "DBSCAN",
            {
                "eps": eps,
                "min_samples": min_samples
            }
        )

        results.append(metrics)

        score = metrics["silhouette_score"]

        print(
            f"eps={eps}, min_samples={min_samples}, "
            f"clusters={metrics['clusters']}, "
            f"silhouette={score}"
        )

        if score is not None and score > best_dbscan_score:

            best_dbscan_score = score
            best_dbscan_model = model
            best_dbscan_labels = labels

            best_dbscan_params = {
                "eps": eps,
                "min_samples": min_samples
            }

# --------------------------------------------------
# 7. SAVE EVALUATION RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_path = os.path.join(
    OUTPUT_DIR,
    "clustering_model_comparison.csv"
)

results_df.to_csv(results_path, index=False)

# --------------------------------------------------
# 8. SAVE BEST K-MEANS CUSTOMER SEGMENTS
# --------------------------------------------------

if best_kmeans_model is not None:

    customers["cluster"] = best_kmeans_labels

    summary = customers.groupby("cluster").agg(
        customer_count=("customer_id", "count"),
        average_recency=("days_since_last_purchase", "mean"),
        average_frequency=("order_count", "mean"),
        average_monetary=("total_spend", "mean"),
        average_order_value=("avg_order_value", "mean")
    ).round(2)

    summary_path = os.path.join(
        OUTPUT_DIR,
        "kmeans_segment_profiles.csv"
    )

    summary.to_csv(summary_path)

    customers.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "kmeans_customer_segments.csv"
        ),
        index=False
    )

    joblib.dump(
        {
            "scaler": scaler,
            "model": best_kmeans_model,
            "features": FEATURES,
            "selected_k": best_k
        },
        os.path.join(
            OUTPUT_DIR,
            "best_kmeans_model.joblib"
        )
    )

# --------------------------------------------------
# 9. SAVE BEST DBSCAN RESULTS
# --------------------------------------------------

if best_dbscan_model is not None:

    dbscan_results = customers.copy()

    dbscan_results["cluster"] = best_dbscan_labels

    dbscan_results.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "dbscan_customer_segments.csv"
        ),
        index=False
    )

# --------------------------------------------------
# 10. SAVE FINAL METRICS
# --------------------------------------------------

final_metrics = {
    "total_customers": int(len(df)),
    "customers_with_purchases": int(len(customers)),
    "customers_without_purchases": int(len(df) - len(customers)),
    "features": FEATURES,
    "best_kmeans_clusters": int(best_k) if best_k is not None else None,
    "best_kmeans_silhouette": (
        float(best_kmeans_score)
        if best_kmeans_model is not None else None
    ),
    "best_dbscan_parameters": best_dbscan_params,
    "best_dbscan_silhouette": (
        float(best_dbscan_score)
        if best_dbscan_model is not None else None
    ),
    "evaluation_sample_size": int(sample_size)
}

with open(
    os.path.join(OUTPUT_DIR, "experiment_metrics.json"),
    "w"
) as file:

    json.dump(final_metrics, file, indent=4)

# --------------------------------------------------
# 11. FINAL REPORT
# --------------------------------------------------

print("\n======================================")
print("RFM SEGMENTATION EXPERIMENT COMPLETED")
print("======================================")

print("\nBest K-Means clusters:", best_k)
print("Best K-Means silhouette:", best_kmeans_score)

print("\nBest DBSCAN parameters:", best_dbscan_params)
print("Best DBSCAN silhouette:", best_dbscan_score)

print("\nSaved files in:", OUTPUT_DIR)

print("\nExperiment completed successfully!")