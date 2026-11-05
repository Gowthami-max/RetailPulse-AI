import os
import json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# ------------------------------------------------
# PATHS
# ------------------------------------------------

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
    "extended_rfm_experiment"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

print("Loading customer data...")

df = pd.read_csv(DATA_PATH)

df = df[df["order_count"] > 0].copy()

df["first_purchase_date"] = pd.to_datetime(
    df["first_purchase_date"]
)

df["last_purchase_date"] = pd.to_datetime(
    df["last_purchase_date"]
)

df["purchase_duration_days"] = (
    df["last_purchase_date"] -
    df["first_purchase_date"]
).dt.days

df["purchase_duration_days"] = (
    df["purchase_duration_days"].fillna(0).clip(lower=0)
)

print("Purchasing customers:", len(df))

# ------------------------------------------------
# FEATURE SETS
# ------------------------------------------------

traditional_features = [
    "days_since_last_purchase",
    "order_count",
    "total_spend"
]

extended_features = [
    "days_since_last_purchase",
    "order_count",
    "total_spend",
    "avg_order_value",
    "purchase_duration_days"
]

feature_sets = {
    "Traditional_RFM": traditional_features,
    "Extended_RFM": extended_features
}

all_results = []

best_models = {}

# ------------------------------------------------
# EXPERIMENT
# ------------------------------------------------

for experiment_name, features in feature_sets.items():

    print("\n================================")
    print("EXPERIMENT:", experiment_name)
    print("================================")

    X = df[features].copy()

    X = np.log1p(X)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    best_score = -1
    best_model = None
    best_labels = None
    best_k = None

    for k in range(2, 9):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = model.fit_predict(X_scaled)

        silhouette = silhouette_score(
            X_scaled,
            labels,
            sample_size=3000,
            random_state=42
        )

        db_score = davies_bouldin_score(
            X_scaled,
            labels
        )

        ch_score = calinski_harabasz_score(
            X_scaled,
            labels
        )

        result = {
            "experiment": experiment_name,
            "features": len(features),
            "clusters": k,
            "silhouette_score": float(silhouette),
            "davies_bouldin_score": float(db_score),
            "calinski_harabasz_score": float(ch_score)
        }

        all_results.append(result)

        print(
            f"K={k} | "
            f"Silhouette={silhouette:.4f} | "
            f"Davies-Bouldin={db_score:.4f} | "
            f"Calinski-Harabasz={ch_score:.2f}"
        )

        if silhouette > best_score:

            best_score = silhouette
            best_model = model
            best_labels = labels
            best_k = k

    best_models[experiment_name] = {
        "model": best_model,
        "scaler": scaler,
        "features": features,
        "labels": best_labels,
        "best_k": best_k,
        "silhouette": best_score
    }

    # Save cluster profiles

    profile_data = df.copy()

    profile_data["cluster"] = best_labels

    profile = profile_data.groupby("cluster").agg(
        customer_count=("customer_id", "count"),
        avg_recency=("days_since_last_purchase", "mean"),
        avg_frequency=("order_count", "mean"),
        avg_monetary=("total_spend", "mean"),
        avg_order_value=("avg_order_value", "mean"),
        avg_purchase_duration=("purchase_duration_days", "mean")
    ).round(2)

    profile.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{experiment_name}_profiles.csv"
        )
    )

    profile_data.to_csv(
        os.path.join(
            OUTPUT_DIR,
            f"{experiment_name}_customers.csv"
        ),
        index=False
    )

# ------------------------------------------------
# SAVE COMPARISON
# ------------------------------------------------

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "extended_rfm_comparison.csv"
    ),
    index=False
)

# ------------------------------------------------
# SAVE SUMMARY
# ------------------------------------------------

summary = {}

for name, result in best_models.items():

    summary[name] = {
        "best_clusters": int(result["best_k"]),
        "silhouette_score": float(result["silhouette"]),
        "features": result["features"]
    }

with open(
    os.path.join(
        OUTPUT_DIR,
        "extended_rfm_metrics.json"
    ),
    "w"
) as file:

    json.dump(summary, file, indent=4)

print("\n================================")
print("EXTENDED RFM EXPERIMENT COMPLETED")
print("================================")

print(json.dumps(summary, indent=4))

print("\nOutput directory:", OUTPUT_DIR)