import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

df = pd.read_csv("../data/processed/customer_analytics_rfm.csv")

df["churn"] = (
    df["days_since_last_purchase"] >= 180
).astype(int)

print("\nFEATURE LEAKAGE INVESTIGATION")
print("=" * 50)

for col in df.columns:

    if col in ["churn", "days_since_last_purchase", "customer_id"]:
        continue

    try:
        if df[col].dtype == "object":
            encoded = pd.factorize(df[col])[0]
            if len(np.unique(encoded)) < 2:
                continue
            score = roc_auc_score(df["churn"], encoded)
        else:
            if df[col].nunique() < 2:
                continue
            score = roc_auc_score(df["churn"], df[col])

        reverse_score = max(score, 1 - score)

        print(f"{col:30s} AUC: {reverse_score:.4f}")

    except Exception:
        pass

print("\nInvestigation completed.")