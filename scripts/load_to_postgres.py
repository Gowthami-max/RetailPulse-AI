import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
import re
import json
from datetime import datetime

# PostgreSQL connection
DATABASE_URL = (
    "postgresql+psycopg2://retailpulse:"
    "retailpulse123@localhost:5432/retailpulse_db"
)

engine = create_engine(DATABASE_URL)

# Dataset folder
DATA_FOLDER = Path("data/processed")

# Log results
results = []

def clean_name(name):
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")

print("\nRETAILPULSE POSTGRESQL DATA LOADER")
print("=" * 50)

for file in DATA_FOLDER.glob("*.csv"):

    table_name = clean_name(file.stem)

    try:
        print(f"\nLoading: {file.name}")

        df = pd.read_csv(file, low_memory=False)

        # Clean column names
        df.columns = [clean_name(col) for col in df.columns]

        # Load into PostgreSQL
        df.to_sql(
            table_name,
            engine,
            if_exists="replace",
            index=False,
            chunksize=1000
        )

        results.append({
            "table": table_name,
            "rows": len(df),
            "status": "SUCCESS"
        })

        print(f"SUCCESS: {len(df)} records loaded")

    except Exception as error:

        results.append({
            "table": table_name,
            "status": "FAILED",
            "error": str(error)
        })

        print(f"FAILED: {error}")

# Save loading report
report_path = Path("models/outputs/postgres_loading_report.json")
report_path.parent.mkdir(parents=True, exist_ok=True)

with open(report_path, "w") as f:
    json.dump(results, f, indent=4)

print("\n" + "=" * 50)
print("POSTGRESQL LOADING COMPLETED")

for item in results:
    print(item)

engine.dispose()