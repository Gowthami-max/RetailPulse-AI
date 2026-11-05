import pandas as pd

orders = pd.read_csv("data/processed/orders.csv")
items = pd.read_csv("data/processed/order_items.csv")

orders["order_date"] = pd.to_datetime(orders["order_date"])

usd = orders[
    (orders["currency"] == "USD") &
    (orders["order_status"].str.lower() == "delivered")
].copy()

usd["week"] = (
    usd["order_date"]
    .dt.to_period("W-SUN")
    .dt.end_time
    .dt.normalize()
)

period = usd[
    usd["week"].isin(
        pd.to_datetime(["2026-05-24", "2026-05-31"])
    )
]

print("\nWEEKLY PRODUCT AND ORDER ANALYSIS")

for week, group in period.groupby("week"):

    print(f"\nWeek: {week.date()}")
    print("Orders:", group["order_id"].nunique())
    print("Customers:", group["customer_id"].nunique())
    print("Average order value:", round(group["order_total"].mean(), 2))

    print("Total items:", group["item_count"].sum())

    print("Revenue:", round(group["order_total"].sum(), 2))