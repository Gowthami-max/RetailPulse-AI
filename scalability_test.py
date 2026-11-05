import os
import time
import json
import gc
import numpy as np
import pandas as pd
from datetime import datetime

# ---------------------------------------
# CONFIGURATION
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(
    BASE_DIR, "models", "outputs", "scalability"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TOTAL_RECORDS = 10_000_000
BATCH_SIZE = 500_000
NUM_BATCHES = TOTAL_RECORDS // BATCH_SIZE

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "scalability_10m_results.csv"
)

rng = np.random.default_rng(42)

# ---------------------------------------
# BATCH PROCESSING
# ---------------------------------------

print("\nRETAILPULSE SCALABILITY BENCHMARK")
print("=" * 55)

print(f"Total transactions: {TOTAL_RECORDS:,}")
print(f"Batch size: {BATCH_SIZE:,}")
print(f"Number of batches: {NUM_BATCHES}")

start_time = time.perf_counter()

total_revenue = 0.0
total_quantity = 0
processed_records = 0

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

for batch_number in range(NUM_BATCHES):

    batch_start = time.perf_counter()

    # Generate synthetic transaction records
    batch = pd.DataFrame({
        "transaction_id": np.arange(
            batch_number * BATCH_SIZE + 1,
            (batch_number + 1) * BATCH_SIZE + 1
        ),
        "customer_id": rng.integers(
            1, 100_001, size=BATCH_SIZE
        ),
        "product_id": rng.integers(
            1, 10_001, size=BATCH_SIZE
        ),
        "quantity": rng.integers(
            1, 6, size=BATCH_SIZE
        ),
        "unit_price": rng.uniform(
            10, 5000, size=BATCH_SIZE
        ).round(2)
    })

    # Retail analytics calculations
    batch["revenue"] = (
        batch["quantity"] * batch["unit_price"]
    )

    # Aggregate batch results
    total_revenue += batch["revenue"].sum()
    total_quantity += batch["quantity"].sum()

    # Save batch
    batch.to_csv(
        OUTPUT_FILE,
        mode="a",
        index=False,
        header=(batch_number == 0)
    )

    processed_records += len(batch)

    elapsed_batch = time.perf_counter() - batch_start

    print(
        f"Batch {batch_number + 1}/{NUM_BATCHES} | "
        f"Records: {processed_records:,} | "
        f"Time: {elapsed_batch:.2f}s"
    )

    del batch
    gc.collect()

# ---------------------------------------
# FINAL RESULTS
# ---------------------------------------

total_time = time.perf_counter() - start_time

file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)

summary = {
    "benchmark": "RetailPulse 10 Million Transaction Test",
    "test_date": datetime.now().isoformat(),
    "total_transactions": processed_records,
    "batch_size": BATCH_SIZE,
    "number_of_batches": NUM_BATCHES,
    "total_revenue": round(float(total_revenue), 2),
    "total_quantity": int(total_quantity),
    "processing_time_seconds": round(total_time, 2),
    "processing_time_minutes": round(total_time / 60, 2),
    "output_file_size_mb": round(file_size_mb, 2),
    "transactions_per_second": round(
        processed_records / total_time, 2
    ),
    "benchmark_environment": "Local Windows machine"
}

SUMMARY_FILE = os.path.join(
    OUTPUT_DIR, "scalability_summary.json"
)

with open(SUMMARY_FILE, "w") as f:
    json.dump(summary, f, indent=4)

print("\n" + "=" * 55)
print("SCALABILITY BENCHMARK RESULTS")
print("=" * 55)

print("Transactions processed:", f"{processed_records:,}")
print("Total processing time:", total_time, "seconds")
print("Processing speed:", summary["transactions_per_second"], "transactions/sec")
print("Output file size:", round(file_size_mb, 2), "MB")
print("Total revenue:", round(total_revenue, 2))

print("\nResults saved in:", OUTPUT_DIR)
print("Scalability test completed!")