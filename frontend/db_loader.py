import json
from io import StringIO

import pandas as pd
import redis
from sqlalchemy import create_engine, text


DATABASE_URL = (
    "postgresql+psycopg2://retailpulse:"
    "retailpulse123@localhost:5432/retailpulse_db"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# Redis connection
cache = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

# Cache smaller, frequently used tables.
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

    # Try Redis first
    if table_name in CACHE_TABLES:
        try:
            cached = cache.get(cache_key)

            if cached:
                print(f"REDIS CACHE HIT: {table_name}")
                return pd.read_json(
                    StringIO(cached),
                    orient="split"
                )
        except Exception:
            pass

    # Load from PostgreSQL
    print(f"POSTGRESQL LOAD: {table_name}")

    query = text(f'SELECT * FROM "{table_name}"')

    with engine.connect() as connection:
        df = pd.read_sql_query(query, connection)

    # Store smaller tables in Redis
    if table_name in CACHE_TABLES:
        try:
            cache.set(
                cache_key,
                df.to_json(orient="split", date_format="iso"),
                ex=300
            )
            print(f"REDIS CACHE UPDATED: {table_name}")
        except Exception:
            pass

    return df


def test_connections():

    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    print("POSTGRESQL: CONNECTED")

    cache.ping()
    print("REDIS: CONNECTED")


if __name__ == "__main__":

    test_connections()

    print("\nTESTING DATA LOADING")

    for table in ["orders", "customers", "products"]:

        df = load_table(table)

        print(
            f"{table}: {len(df)} records, "
            f"{len(df.columns)} columns"
        )

    print("\nDATABASE LOADER TEST COMPLETED")

    engine.dispose()