from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.config import (
    APP_NAME,
    APP_VERSION,
    TRANSACTIONS_FILE,
    CHURN_FILE,
    INVENTORY_FILE
)

from backend.data_loader import (
    load_transactions,
    filter_transactions,
    sales_rows,
    clear_data_cache
)

from backend.analytics import (
    summary_metrics,
    monthly_sales,
    daily_sales,
    top_products,
    country_sales,
    customer_summary,
    order_summary
)


# =====================================
# APPLICATION
# =====================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="AI-Powered Retail Analytics Platform"
)


# =====================================
# CORS
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =====================================
# GLOBAL FILTER FUNCTION
# =====================================

def get_filtered_data(
    from_date,
    to_date,
    customer_id,
    country
):

    if from_date and to_date and from_date > to_date:
        raise HTTPException(
            status_code=422,
            detail="Invalid date range"
        )

    try:

        df = load_transactions()

        return filter_transactions(
            df,
            from_date,
            to_date,
            customer_id,
            country
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=503,
            detail=str(error)
        )

    except ValueError as error:

        raise HTTPException(
            status_code=422,
            detail=str(error)
        )


# =====================================
# HEALTH CHECK
# =====================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "application": APP_NAME,
        "version": APP_VERSION
    }


# =====================================
# ROOT
# =====================================

@app.get("/")
def root():

    return {
        "message": "Welcome to RetailPulse API",
        "documentation": "/docs",
        "health": "/health"
    }


# =====================================
# DATASET METADATA
# =====================================

@app.get("/metadata")
def metadata():

    df = load_transactions()

    dates = df["InvoiceDate"].dropna()

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),

        "customers": int(
            df["Customer ID"].nunique()
        ),

        "countries": int(
            df["Country"].nunique()
        ),

        "products": int(
            df["StockCode"].nunique()
        ),

        "date_min": (
            None if dates.empty
            else dates.min().date().isoformat()
        ),

        "date_max": (
            None if dates.empty
            else dates.max().date().isoformat()
        )
    }


# =====================================
# GLOBAL FILTER OPTIONS
# =====================================

@app.get("/filters")
def filter_options():

    df = load_transactions()

    dates = df["InvoiceDate"].dropna()

    return {
        "countries": sorted(
            df["Country"].dropna().unique().tolist()
        ),

        "customer_count": int(
            df["Customer ID"].nunique()
        ),

        "minimum_date": (
            None if dates.empty
            else dates.min().date().isoformat()
        ),

        "maximum_date": (
            None if dates.empty
            else dates.max().date().isoformat()
        )
    }


# =====================================
# EXECUTIVE SUMMARY
# =====================================

@app.get("/analytics/summary")
def summary(

    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    customer_id: Optional[str] = None,
    country: Optional[str] = None

):

    df = get_filtered_data(
        from_date,
        to_date,
        customer_id,
        country
    )

    return summary_metrics(df)


# =====================================
# SALES TREND
# =====================================

@app.get("/analytics/sales-trend")
def sales_trend(

    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    customer_id: Optional[str] = None,
    country: Optional[str] = None,

    granularity: str = Query(
        "month",
        pattern="^(day|month)$"
    )

):

    df = get_filtered_data(
        from_date,
        to_date,
        customer_id,
        country
    )

    if granularity == "day":
        result = daily_sales(df)
    else:
        result = monthly_sales(df)

    return {
        "granularity": granularity,
        "data": result
    }


# =====================================
# PRODUCT ANALYTICS
# =====================================

@app.get("/analytics/products")
def products(

    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    customer_id: Optional[str] = None,
    country: Optional[str] = None,

    limit: int = Query(10, ge=1, le=100)

):

    df = get_filtered_data(
        from_date,
        to_date,
        customer_id,
        country
    )

    return {
        "data": top_products(df, limit)
    }


# =====================================
# COUNTRY ANALYTICS
# =====================================

@app.get("/analytics/countries")
def countries(

    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    customer_id: Optional[str] = None,
    country: Optional[str] = None,

    limit: int = Query(50, ge=1, le=100)

):

    df = get_filtered_data(
        from_date,
        to_date,
        customer_id,
        country
    )

    return {
        "data": country_sales(df, limit)
    }


# =====================================
# CUSTOMER ANALYTICS
# =====================================

@app.get("/analytics/customers")
def customers(

    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    customer_id: Optional[str] = None,
    country: Optional[str] = None,

    limit: int = Query(20, ge=1, le=100)

):

    df = get_filtered_data(
        from_date,
        to_date,
        customer_id,
        country
    )

    return {
        "data": customer_summary(df, limit)
    }


# =====================================
# INDIVIDUAL CUSTOMER
# =====================================

@app.get("/analytics/customer/{customer_id}")
def individual_customer(

    customer_id: str,

    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    country: Optional[str] = None

):

    df = get_filtered_data(
        from_date,
        to_date,
        customer_id,
        country
    )

    if df.empty:

        return {
            "found": False,
            "customer_id": customer_id,
            "metrics": summary_metrics(df),
            "products": [],
            "monthly_sales": []
        }

    return {
        "found": True,
        "customer_id": customer_id,
        "metrics": summary_metrics(df),
        "products": top_products(df),
        "monthly_sales": monthly_sales(df)
    }


# =====================================
# ORDER ANALYTICS
# =====================================

@app.get("/analytics/orders")
def orders(

    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    customer_id: Optional[str] = None,
    country: Optional[str] = None,

    limit: int = Query(100, ge=1, le=500)

):

    df = get_filtered_data(
        from_date,
        to_date,
        customer_id,
        country
    )

    return {
        "data": order_summary(df, limit)
    }


# =====================================
# RETURNS AND CANCELLATIONS
# =====================================

@app.get("/analytics/returns")
def returns(

    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    customer_id: Optional[str] = None,
    country: Optional[str] = None

):

    df = get_filtered_data(
        from_date,
        to_date,
        customer_id,
        country
    )

    cancelled = df[df["IsCancelled"]]

    negative = df[df["IsNegativeQuantity"]]

    return {
        "cancelled_rows": int(len(cancelled)),

        "cancelled_invoices": int(
            cancelled["Invoice"].nunique()
        ),

        "negative_quantity_rows": int(len(negative)),

        "negative_quantity_value": round(
            float(negative["TotalAmount"].sum()), 2
        )
    }


# =====================================
# RFM CUSTOMER ANALYTICS
# =====================================

@app.get("/analytics/rfm")
def rfm(

    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    customer_id: Optional[str] = None,
    country: Optional[str] = None

):

    import pandas as pd

    df = get_filtered_data(
        from_date,
        to_date,
        customer_id,
        country
    )

    sales = sales_rows(df)

    sales = sales[
        sales["Customer ID"].notna()
    ]

    if sales.empty:
        return {"data": []}

    reference_date = (
        sales["InvoiceDate"].max()
        + pd.Timedelta(days=1)
    )

    result = (
        sales.groupby("Customer ID")
        .agg(
            last_purchase=("InvoiceDate", "max"),
            frequency=("Invoice", "nunique"),
            monetary=("TotalAmount", "sum")
        )
        .reset_index()
    )

    result["recency_days"] = (
        reference_date - result["last_purchase"]
    ).dt.days

    result["Customer ID"] = (
        result["Customer ID"]
        .astype(int)
        .astype(str)
    )

    result["last_purchase"] = (
        result["last_purchase"]
        .dt.strftime("%Y-%m-%d")
    )

    result["monetary"] = result["monetary"].round(2)

    return {
        "data": result.to_dict(orient="records")
    }


# =====================================
# CHURN DATA AVAILABILITY
# =====================================

@app.get("/analytics/churn-source")
def churn_source():

    if not CHURN_FILE.exists():

        return {
            "available": False,
            "message": "Churn source file not found"
        }

    import pandas as pd

    df = pd.read_csv(
        CHURN_FILE,
        nrows=5
    )

    return {
        "available": True,
        "columns": list(df.columns),
        "message": "Existing inactivity-risk data. Model validation is still required."
    }


# =====================================
# INVENTORY DATA AVAILABILITY
# =====================================

@app.get("/analytics/inventory-source")
def inventory_source():

    if not INVENTORY_FILE.exists():

        return {
            "available": False,
            "message": "Inventory source file not found"
        }

    import pandas as pd

    df = pd.read_csv(
        INVENTORY_FILE,
        nrows=5
    )

    return {
        "available": True,
        "columns": list(df.columns),
        "message": "Historical demand-risk output, not live stock information."
    }


# =====================================
# RELOAD DATASET
# =====================================

@app.post("/admin/reload-data")
def reload_data():

    clear_data_cache()

    df = load_transactions()

    return {
        "status": "reloaded",
        "rows": len(df)
    }