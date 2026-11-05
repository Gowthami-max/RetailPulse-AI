from pathlib import Path
import os

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset directory
DATA_DIR = Path(
    os.getenv(
        "RETAILPULSE_DATA_DIR",
        BASE_DIR / "data" / "processed"
    )
)

# Dataset files
TRANSACTIONS_FILE = DATA_DIR / "cleaned_all_transactions.csv"

RFM_FILE = DATA_DIR / "rfm_customer_segments.csv"

CHURN_FILE = DATA_DIR / "customer_inactivity_risk.csv"

INVENTORY_FILE = DATA_DIR / "inventory_optimization_results.csv"

# Application configuration
APP_NAME = "RetailPulse API"
APP_VERSION = "1.0.0"