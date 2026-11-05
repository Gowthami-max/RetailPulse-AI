import pandas as pd

df = pd.read_csv("data/processed/orders.csv")

df["order_date"] = pd.to_datetime(df["order_date"])

usd = df[
    (df["currency"] == "USD") &
    (df["order_status"].str.lower() == "delivered")
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

print("\nSALES CHANNEL COMPARISON")

channel = period.groupby(
    ["week", "sales_channel"]
).agg(
    Revenue=("order_total", "sum"),
    Orders=("order_id", "nunique"),
    Customers=("customer_id", "nunique")
).reset_index()

print(channel.to_string(index=False))

print("\nCAMPAIGN COMPARISON")

campaign = period.groupby(
    ["week", "campaign_id"]
).agg(
    Revenue=("order_total", "sum"),
    Orders=("order_id", "nunique")
).reset_index()

print(campaign.to_string(index=False))