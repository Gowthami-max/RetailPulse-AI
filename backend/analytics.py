import pandas as pd

from backend.data_loader import sales_rows


# =====================================
# EXECUTIVE SUMMARY
# =====================================

def summary_metrics(df):

    sales = sales_rows(df)

    return {
        "total_revenue": round(
            float(sales["TotalAmount"].sum()), 2
        ),

        "total_orders": int(
            sales["Invoice"].nunique()
        ),

        "total_customers": int(
            sales["Customer ID"].nunique()
        ),

        "total_products": int(
            sales["StockCode"].nunique()
        ),

        "total_transactions": int(len(df)),

        "cancelled_rows": int(
            df["IsCancelled"].sum()
        ),

        "return_rows": int(
            df["IsNegativeQuantity"].sum()
        ),

        "revenue_definition":
            "Positive completed transaction value. Not profit."
    }


# =====================================
# MONTHLY SALES
# =====================================

def monthly_sales(df):

    sales = sales_rows(df)

    if sales.empty:
        return []

    sales["Month"] = (
        sales["InvoiceDate"]
        .dt.to_period("M")
        .astype(str)
    )

    result = (
        sales.groupby("Month")
        .agg(
            revenue=("TotalAmount", "sum"),
            orders=("Invoice", "nunique"),
            quantity=("Quantity", "sum")
        )
        .reset_index()
        .sort_values("Month")
    )

    result["revenue"] = result["revenue"].round(2)

    return result.to_dict(orient="records")


# =====================================
# DAILY SALES
# =====================================

def daily_sales(df):

    sales = sales_rows(df)

    if sales.empty:
        return []

    sales["Date"] = (
        sales["InvoiceDate"]
        .dt.strftime("%Y-%m-%d")
    )

    result = (
        sales.groupby("Date")
        .agg(
            revenue=("TotalAmount", "sum"),
            orders=("Invoice", "nunique")
        )
        .reset_index()
        .sort_values("Date")
    )

    result["revenue"] = result["revenue"].round(2)

    return result.to_dict(orient="records")


# =====================================
# PRODUCT PERFORMANCE
# =====================================

def top_products(df, limit=10):

    sales = sales_rows(df)

    if sales.empty:
        return []

    result = (
        sales.groupby(
            ["StockCode", "Description"],
            dropna=False
        )
        .agg(
            revenue=("TotalAmount", "sum"),
            quantity=("Quantity", "sum"),
            orders=("Invoice", "nunique")
        )
        .reset_index()
        .sort_values(
            "revenue",
            ascending=False
        )
        .head(limit)
    )

    result["revenue"] = result["revenue"].round(2)

    return result.to_dict(orient="records")


# =====================================
# COUNTRY ANALYSIS
# =====================================

def country_sales(df, limit=50):

    sales = sales_rows(df)

    if sales.empty:
        return []

    result = (
        sales.groupby("Country")
        .agg(
            revenue=("TotalAmount", "sum"),
            orders=("Invoice", "nunique"),
            customers=("Customer ID", "nunique")
        )
        .reset_index()
        .sort_values(
            "revenue",
            ascending=False
        )
        .head(limit)
    )

    result["revenue"] = result["revenue"].round(2)

    return result.to_dict(orient="records")


# =====================================
# CUSTOMER ANALYTICS
# =====================================

def customer_summary(df, limit=20):

    sales = sales_rows(df)

    sales = sales[
        sales["Customer ID"].notna()
    ]

    if sales.empty:
        return []

    result = (
        sales.groupby("Customer ID")
        .agg(
            revenue=("TotalAmount", "sum"),
            orders=("Invoice", "nunique"),
            quantity=("Quantity", "sum"),
            last_purchase=("InvoiceDate", "max")
        )
        .reset_index()
        .sort_values(
            "revenue",
            ascending=False
        )
        .head(limit)
    )

    result["Customer ID"] = (
        result["Customer ID"]
        .astype(int)
        .astype(str)
    )

    result["revenue"] = result["revenue"].round(2)

    result["last_purchase"] = (
        result["last_purchase"]
        .dt.strftime("%Y-%m-%d")
    )

    return result.to_dict(orient="records")


# =====================================
# ORDER ANALYSIS
# =====================================

def order_summary(df, limit=100):

    if df.empty:
        return []

    result = (
        df.groupby("Invoice")
        .agg(
            date=("InvoiceDate", "min"),
            amount=("TotalAmount", "sum"),
            line_items=("StockCode", "count"),
            customer_id=("Customer ID", "first"),
            country=("Country", "first"),
            cancelled=("IsCancelled", "max")
        )
        .reset_index()
        .sort_values(
            "date",
            ascending=False
        )
        .head(limit)
    )

    result["date"] = (
        result["date"]
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    result["amount"] = result["amount"].round(2)

    result["customer_id"] = result["customer_id"].apply(
        lambda x: None if pd.isna(x) else str(int(x))
    )

    return result.to_dict(orient="records")