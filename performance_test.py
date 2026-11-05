import os
import time
import pandas as pd
import json
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "processed")
OUTPUT = os.path.join(BASE, "models", "outputs")

os.makedirs(OUTPUT, exist_ok=True)

files = [
    "orders.csv",
    "order_items.csv",
    "customers.csv",
    "products.csv",
    "payments.csv",
    "shipments.csv",
    "returns.csv",
    "inventory_snapshot.csv"
]

results = []
total_start = time.perf_counter()

print("\nRETAILPULSE PERFORMANCE TEST")
print("=" * 50)

for filename in files:

    filepath = os.path.join(DATA, filename)

    if not os.path.exists(filepath):
        print(f"SKIPPED: {filename} not found")
        continue

    start = time.perf_counter()

    try:
        df = pd.read_csv(filepath)

        # Basic processing
        df = df.drop_duplicates()

        elapsed = time.perf_counter() - start

        results.append({
            "File": filename,
            "Rows": len(df),
            "Columns": len(df.columns),
            "Processing_Time_Seconds": round(elapsed, 4),
            "Status": "SUCCESS"
        })

        print(
            f"{filename}: {len(df):,} rows | "
            f"{elapsed:.3f} seconds"
        )

    except Exception as e:

        results.append({
            "File": filename,
            "Rows": 0,
            "Columns": 0,
            "Processing_Time_Seconds": 0,
            "Status": str(e)
        })

total_time = time.perf_counter() - total_start

result_df = pd.DataFrame(results)

result_df.to_csv(
    os.path.join(OUTPUT, "performance_test_results.csv"),
    index=False
)

summary = {
    "test_name": "RetailPulse CSV Processing Benchmark",
    "test_date": datetime.now().isoformat(),
    "total_files_processed": len(results),
    "total_processing_time_seconds": round(total_time, 3),
    "total_processing_time_minutes": round(total_time / 60, 3),
    "target_seconds": 300,
    "target_met": total_time < 300
}

with open(
    os.path.join(OUTPUT, "performance_summary.json"),
    "w"
) as f:
    json.dump(summary, f, indent=4)

print("\n" + "=" * 50)
print("PERFORMANCE TEST SUMMARY")
print("=" * 50)

print("Files processed:", len(results))
print("Total time:", round(total_time, 2), "seconds")
print("Total time:", round(total_time / 60, 2), "minutes")

if total_time < 300:
    print("RESULT: UNDER 5 MINUTES")
else:
    print("RESULT: TARGET NOT MET")

print("\nResults saved in models/outputs/")