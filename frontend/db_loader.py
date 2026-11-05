import os
import json
from io import StringIO

import pandas as pd
import redis
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv(override=True)

# Choose database: local or cloud
USE_NEON = os.getenv("USE_NEON", "false").lower() == "true"

if USE_NEON:
    DATABASE_URL = os.getenv("NEON_DATABASE_URL")
    print("DATABASE MODE: NEON CLOUD")
else:
    DATABASE_URL = os.getenv("DATABASE_URL")
    print("DATABASE MODE: LOCAL POSTGRESQL")

if not DATABASE_URL:
    raise ValueError("Database URL missing in .env")

if USE_NEON:
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://",
        "postgresql+psycopg2://",
        1
    )

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

# Redis connection
cache = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

CACHE_TABLES = {
    "customers",
    "products",
    "customer_analytics_rfm",
    "product_performance",
    "country_sales_summary",
    "daily_sales_summary",
    "monthly_sales_summary",
    "payment_summary",
    "inventory_snapshot",
    "campaigns",
}

def load_table(table_name):

    cache_key = f"retailpulse:table:{table_name}"

    if table_name in CACHE_TABLES:
        try:
            cached = cache.get(cache_key)

            if cached:
                return pd.read_json(
                    StringIO(cached),
                    orient="split"
                )
        except Exception:
            pass

    print(f"POSTGRESQL LOAD: {table_name}")

    query = text(f'SELECT * FROM "{table_name}"')

    with engine.connect() as connection:
        df = pd.read_sql_query(query, connection)

    if table_name in CACHE_TABLES:
        try:
            cache.set(
                cache_key,
                df.to_json(
                    orient="split",
                    date_format="iso"
                ),
                ex=300
            )
        except Exception:
            pass

    return df


def test_connections():

    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    print("POSTGRESQL: CONNECTED")

    try:
        cache.ping()
        print("REDIS: CONNECTED")
    except Exception:
        print("REDIS: NOT AVAILABLE - continuing without cache")


if __name__ == "__main__":

    test_connections()

    print("\nTESTING DATA LOADING")

    for table in ["orders", "customers", "products"]:
        df = load_table(table)
        print(f"{table}: {len(df)} records, {len(df.columns)} columns")

    print("\nDATABASE LOADER TEST COMPLETED")

    engine.dispose()