import pandas as pd
from functools import lru_cache

from backend.config import TRANSACTIONS_FILE


REQUIRED_COLUMNS = [
    "Invoice",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Customer ID",
    "Country"
]


@lru_cache(maxsize=1)
def load_transactions():

    if not TRANSACTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {TRANSACTIONS_FILE}"
        )

    print("Loading RetailPulse dataset...")

    df = pd.read_csv(
        TRANSACTIONS_FILE,
        low_memory=False
    )

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"],
        errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"],
        errors="coerce"
    ).fillna(0)

    df["Price"] = pd.to_numeric(
        df["Price"],
        errors="coerce"
    ).fillna(0)

    if "TotalAmount" not in df.columns:
        df["TotalAmount"] = (
            df["Quantity"] * df["Price"]
        )

    # Validate transaction flags
    flag_defaults = {
        "IsCancelled": df["Invoice"].astype(str).str.startswith("C"),
        "IsNegativeQuantity": df["Quantity"] < 0,
        "IsInvalidPrice": df["Price"] < 0
    }

    for column, default in flag_defaults.items():

        if column not in df.columns:
            df[column] = default

        elif df[column].dtype != bool:
            df[column] = (
                df[column]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
            )

    print(f"Loaded {len(df):,} transactions.")

    return df


def filter_transactions(
    df,
    from_date=None,
    to_date=None,
    customer_id=None,
    country=None
):

    filtered = df

    if from_date is not None:

        filtered = filtered[
            filtered["InvoiceDate"] >= pd.Timestamp(from_date)
        ]

    if to_date is not None:

        end_date = (
            pd.Timestamp(to_date)
            + pd.Timedelta(days=1)
        )

        filtered = filtered[
            filtered["InvoiceDate"] < end_date
        ]

    if customer_id is not None:

        filtered = filtered[
            filtered["Customer ID"] == float(customer_id)
        ]

    if country:

        filtered = filtered[
            filtered["Country"] == country
        ]

    return filtered.copy()


def sales_rows(df):

    mask = (
        (~df["IsCancelled"])
        & (~df["IsNegativeQuantity"])
        & (~df["IsInvalidPrice"])
        & (df["Quantity"] > 0)
        & (df["Price"] >= 0)
    )

    return df.loc[mask].copy()


def clear_data_cache():

    load_transactions.cache_clear()