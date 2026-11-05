import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text

# Load environment variables
load_dotenv(override=True)

NEON_URL = os.getenv("NEON_DATABASE_URL")

if not NEON_URL:
    raise ValueError("NEON_DATABASE_URL missing in .env")

if "YOUR_PASSWORD" in NEON_URL or "paste_your" in NEON_URL:
    raise ValueError("Replace the placeholder with your actual Neon connection string.")

# Use psycopg2 driver
NEON_URL = NEON_URL.replace(
    "postgresql://",
    "postgresql+psycopg2://",
    1
)

engine = create_engine(NEON_URL, pool_pre_ping=True)

DATA_FOLDER = Path("data/processed")

REQUIRED_FILES = [
    "customers.csv",
    "products.csv",
    "orders.csv",
    "order_items.csv",
    "payments.csv",
    "shipments.csv",
    "returns.csv",
    "inventory_snapshot.csv",
    "campaigns.csv",
    "transactions_flat.csv",
    "customer_analytics_rfm.csv",
    "product_performance.csv",
    "country_sales_summary.csv",
    "daily_sales_summary.csv",
    "monthly_sales_summary.csv",
    "payment_summary.csv",
    "live_transactions_seed.csv",
]

def migrate():

    print("\nRETAILPULSE - NEON MIGRATION")
    print("=" * 45)

    # Check all source files exist
    for filename in REQUIRED_FILES:
        if not (DATA_FOLDER / filename).exists():
            raise FileNotFoundError(f"Missing file: {filename}")

    # One transaction: if any table fails, all changes roll back
    with engine.begin() as connection:

        inspector = inspect(connection)

        existing = [
            Path(filename).stem
            for filename in REQUIRED_FILES
            if inspector.has_table(
                Path(filename).stem,
                schema="public"
            )
        ]

        # Never overwrite existing Neon tables
        if existing:
            raise RuntimeError(
                f"Migration stopped. These tables already exist: {existing}"
            )

        for filename in REQUIRED_FILES:

            file_path = DATA_FOLDER / filename
            table_name = Path(filename).stem

            print(f"\nLoading: {table_name}")

            df = pd.read_csv(file_path, low_memory=False)

            # Standardize column names
            df.columns = [
                str(col).strip().lower()
                for col in df.columns
            ]

            if not df.columns.is_unique:
                raise ValueError(
                    f"Duplicate column names in {filename}"
                )

            df.to_sql(
                table_name,
                connection,
                schema="public",
                if_exists="fail",
                index=False,
                chunksize=1000
            )

            # Verify uploaded row count
            uploaded = connection.execute(
                text(f'SELECT COUNT(*) FROM public."{table_name}"')
            ).scalar()

            if uploaded != len(df):
                raise RuntimeError(
                    f"Row count mismatch in {table_name}"
                )

            print(
                f"SUCCESS: {table_name} | "
                f"{uploaded:,} rows | {len(df.columns)} columns"
            )

    print("\n" + "=" * 45)
    print("ALL 17 TABLES MIGRATED SUCCESSFULLY!")
    print("Neon migration completed.")

if __name__ == "__main__":
    try:
        migrate()
    except Exception as error:
        print("\nMIGRATION FAILED:")
        print(error)
        print("No partial migration should remain.")
    finally:
        engine.dispose()