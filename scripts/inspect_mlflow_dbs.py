import sqlite3
from pathlib import Path

files = [
    Path("models/mlflow.db"),
    Path("mlflow.db")
]

for file in files:
    print("\n" + "=" * 50)
    print("DATABASE:", file.resolve())

    if not file.exists():
        print("FILE NOT FOUND")
        continue

    connection = sqlite3.connect(
        f"file:{file.resolve().as_posix()}?mode=ro",
        uri=True
    )

    tables = [
        row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    ]

    print("TABLES:", tables)

    if "experiments" in tables:
        experiments = connection.execute(
            "SELECT experiment_id, name, lifecycle_stage FROM experiments"
        ).fetchall()

        print("EXPERIMENTS:", experiments)

    if "runs" in tables:
        count = connection.execute(
            "SELECT COUNT(*) FROM runs"
        ).fetchone()[0]

        print("TOTAL RUNS:", count)

    connection.close()

print("\nDATABASE INSPECTION COMPLETED")