import pandas as pd

path = "data/processed/orders.csv"

df = pd.read_csv(path)

df["order_date"] = pd.to_datetime(df["order_date"])
df["currency"] = df["currency"].astype(str).str.upper()

# Filter USD delivered orders
usd = df[
    (df["currency"] == "USD") &
    (df["order_status"].astype(str).str.lower() == "delivered")
].copy()

# Weekly revenue
usd["week"] = (
    usd["order_date"]
    .dt.to_period("W-SUN")
    .dt.end_time
    .dt.normalize()
)

weekly = (
    usd.groupby("week")["order_total"]
    .sum()
    .reset_index()
)

# Focus on 2026
weekly = weekly[
    (weekly["week"] >= "2026-01-01") &
    (weekly["week"] <= "2026-11-01")
].copy()

weekly["change_percent"] = (
    weekly["order_total"].pct_change() * 100
)

print("\nUSD WEEKLY REVENUE ANALYSIS")
print(weekly.to_string(index=False))

print("\nLARGEST WEEKLY REVENUE CHANGES")

largest = weekly.loc[
    weekly["change_percent"].abs().nlargest(15).index
]

print(largest.to_string(index=False))