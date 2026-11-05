import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

df = pd.read_csv(DATA_PATH)
df = df[df["order_count"] > 0].copy()

df["first_purchase_date"] = pd.to_datetime(df["first_purchase_date"])
df["last_purchase_date"] = pd.to_datetime(df["last_purchase_date"])

df["purchase_duration_days"] = (
    df["last_purchase_date"] - df["first_purchase_date"]
).dt.days.clip(lower=0)

feature_sets = {
    "Traditional_RFM": [
        "days_since_last_purchase",
        "order_count",
        "total_spend"
    ],
    "Extended_RFM": [
        "days_since_last_purchase",
        "order_count",
        "total_spend",
        "avg_order_value",
        "purchase_duration_days"
    ]
}

all_profiles = []

for name, features in feature_sets.items():

    X = np.log1p(df[features].fillna(0))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for k in [6, 7, 8]:

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

        profile_data = df.copy()
        profile_data["cluster"] = labels

        profile = profile_data.groupby("cluster").agg(
            customer_count=("customer_id", "count"),
            avg_recency=("days_since_last_purchase", "mean"),
            avg_frequency=("order_count", "mean"),
            avg_spending=("total_spend", "mean"),
            avg_order_value=("avg_order_value", "mean"),
            avg_purchase_duration=("purchase_duration_days", "mean")
        ).round(2)

        profile["experiment"] = name
        profile["k"] = k
        profile["silhouette_score"] = round(silhouette, 4)

        all_profiles.append(profile.reset_index())

        print(f"\n{name} | K={k} | Silhouette={silhouette:.4f}")
        print(profile.to_string())

        profile_data.to_csv(
            os.path.join(
                OUTPUT_DIR,
                f"{name}_K{k}_customers.csv"
            ),
            index=False
        )

final_profiles = pd.concat(
    all_profiles,
    ignore_index=True
)

final_profiles.to_csv(
    os.path.join(OUTPUT_DIR, "six_to_eight_cluster_profiles.csv"),
    index=False
)

print("\n================================")
print("CLUSTER PROFILE INSPECTION DONE")
print("================================")