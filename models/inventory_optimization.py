import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "inventory_snapshot.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "models", "outputs", "inventory"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n====================================")
print(" RETAILPULSE INVENTORY OPTIMIZATION")
print("====================================")

# Load inventory
df = pd.read_csv(DATA_PATH)

# Calculate days of cover again
df["days_of_cover"] = np.where(
    df["avg_daily_units_sold_30d"] > 0,
    df["stock_on_hand"] / df["avg_daily_units_sold_30d"],
    np.inf
)

# Inventory risk classification
def classify_inventory(days):

    if days <= 7:
        return "Critical"
    elif days <= 14:
        return "Low Stock"
    elif days <= 30:
        return "Moderate"
    else:
        return "Healthy"


df["inventory_risk"] = df["days_of_cover"].apply(
    classify_inventory
)

# Reorder decision
df["reorder_required"] = (
    (df["stock_on_hand"] <= df["reorder_level"]) |
    (df["days_of_cover"] <= 14)
)

# Target stock coverage assumption: 30 days
TARGET_COVER_DAYS = 30

df["target_stock"] = np.ceil(
    df["avg_daily_units_sold_30d"] * TARGET_COVER_DAYS
).astype(int)

# Recommended reorder quantity
df["recommended_reorder_quantity"] = np.where(
    df["reorder_required"],
    np.maximum(
        0,
        df["target_stock"] - df["stock_on_hand"]
    ),
    0
)

# Estimated reorder cost
df["estimated_reorder_cost"] = (
    df["recommended_reorder_quantity"] * df["unit_cost"]
).round(2)

# Recommendation
df["recommendation"] = np.where(
    df["reorder_required"],
    "Place Reorder",
    "No Immediate Reorder"
)

# Sort by priority
priority = {
    "Critical": 1,
    "Low Stock": 2,
    "Moderate": 3,
    "Healthy": 4
}

df["priority"] = df["inventory_risk"].map(priority)

df = df.sort_values(
    ["priority", "days_of_cover"],
    ascending=[True, True]
)

# Save detailed inventory results
output_path = os.path.join(
    OUTPUT_DIR, "inventory_optimization_results.csv"
)

df.to_csv(output_path, index=False)

# Inventory summary
summary = (
    df.groupby("inventory_risk")
    .agg(
        product_count=("product_id", "count"),
        total_stock=("stock_on_hand", "sum"),
        inventory_value=("inventory_value", "sum"),
        reorder_products=("reorder_required", "sum")
    )
    .reset_index()
)

summary_path = os.path.join(
    OUTPUT_DIR, "inventory_risk_summary.csv"
)

summary.to_csv(summary_path, index=False)

# Reorder report
reorder_report = df[
    df["reorder_required"]
][[
    "product_id",
    "product_name",
    "category",
    "stock_on_hand",
    "reorder_level",
    "days_of_cover",
    "recommended_reorder_quantity",
    "unit_cost",
    "estimated_reorder_cost",
    "inventory_risk"
]]

reorder_path = os.path.join(
    OUTPUT_DIR, "reorder_recommendations.csv"
)

reorder_report.to_csv(reorder_path, index=False)

# Print results
print("\nINVENTORY RISK SUMMARY")
print(summary.to_string(index=False))

print("\nTOTAL PRODUCTS:", len(df))

print(
    "PRODUCTS REQUIRING REORDER:",
    int(df["reorder_required"].sum())
)

print(
    "TOTAL ESTIMATED REORDER COST:",
    round(df["estimated_reorder_cost"].sum(), 2)
)

print("\nFILES SAVED SUCCESSFULLY")
print(output_path)
print(summary_path)
print(reorder_path)

print("\nInventory optimization completed!")