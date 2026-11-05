import os
import json
import joblib
import numpy as np
import pandas as pd

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "orders.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "models", "outputs", "forecasting"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

print("Loading retail orders dataset...")

df = pd.read_csv(DATA_PATH)

df["order_date"] = pd.to_datetime(
    df["order_date"], errors="coerce"
)

df["order_total"] = pd.to_numeric(
    df["order_total"], errors="coerce"
)

df = df.dropna(
    subset=["order_date", "order_total", "currency"]
)

print("Total orders:", len(df))


# ==========================================
# 3. FILTER VALID SALES
# ==========================================

df = df[
    df["order_status"].str.lower() == "delivered"
].copy()

print("Delivered orders:", len(df))


# ==========================================
# 4. FORECASTING CONFIGURATION
# ==========================================

CURRENCIES = ["INR", "USD"]

FORECAST_DAYS = 30
TEST_DAYS = 30

all_forecasts = []
all_test_predictions = []
all_metrics = {}


# ==========================================
# 5. TRAIN SEPARATE CURRENCY MODELS
# ==========================================

for currency in CURRENCIES:

    print("\n" + "=" * 50)
    print(f"TRAINING {currency} FORECAST MODEL")
    print("=" * 50)

    currency_df = df[
        df["currency"] == currency
    ].copy()

    # Aggregate daily revenue
    daily_sales = (
        currency_df.groupby("order_date")["order_total"]
        .sum()
        .reset_index()
    )

    daily_sales.columns = ["ds", "y"]

    # Fill missing dates with zero revenue
    daily_sales = (
        daily_sales.set_index("ds")
        .asfreq("D", fill_value=0)
        .reset_index()
    )

    daily_sales["y"] = daily_sales["y"].astype(float)

    print("Historical days:", len(daily_sales))
    print("Currency:", currency)

    if len(daily_sales) <= TEST_DAYS + 90:
        print("Insufficient data. Skipping currency.")
        continue

    # ======================================
    # 6. TRAIN / TEST SPLIT
    # ======================================

    train = daily_sales.iloc[:-TEST_DAYS].copy()

    test = daily_sales.iloc[-TEST_DAYS:].copy()

    print("Training records:", len(train))
    print("Testing records:", len(test))

    # ======================================
    # 7. TRAIN PROPHET MODEL
    # ======================================

    print("\nTraining Prophet model...")

    evaluation_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.95
    )

    evaluation_model.fit(train)

    # Predict testing period
    test_future = test[["ds"]].copy()

    test_forecast = evaluation_model.predict(
        test_future
    )

    actual = test["y"].values

    predicted = np.maximum(
        test_forecast["yhat"].values, 0
    )

    # ======================================
    # 8. MODEL EVALUATION
    # ======================================

    mae = mean_absolute_error(
        actual, predicted
    )

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    nonzero = actual != 0

    if nonzero.any():
        mape = np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
    else:
        mape = None

    print("\nMODEL PERFORMANCE")
    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print(
        "MAPE:",
        round(mape, 2) if mape is not None else "N/A"
    )

    # Save test predictions
    test_results = pd.DataFrame({
        "date": test["ds"].values,
        "currency": currency,
        "actual_revenue": actual,
        "predicted_revenue": predicted
    })

    all_test_predictions.append(test_results)

    # ======================================
    # 9. TRAIN FINAL MODEL
    # ======================================

    print("\nTraining final model on full history...")

    final_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.95
    )

    final_model.fit(daily_sales)

    # ======================================
    # 10. GENERATE 30-DAY FORECAST
    # ======================================

    future = final_model.make_future_dataframe(
        periods=FORECAST_DAYS,
        freq="D"
    )

    forecast = final_model.predict(future)

    future_forecast = forecast[
        forecast["ds"] > daily_sales["ds"].max()
    ].copy()

    future_forecast["yhat"] = (
        future_forecast["yhat"].clip(lower=0)
    )

    future_forecast["currency"] = currency

    future_forecast = future_forecast[
        [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper",
            "currency"
        ]
    ]

    all_forecasts.append(future_forecast)

    # ======================================
    # 11. SAVE MODEL
    # ======================================

    model_path = os.path.join(
        OUTPUT_DIR,
        f"prophet_{currency}.joblib"
    )

    joblib.dump(final_model, model_path)

    # Save evaluation
    all_metrics[currency] = {
        "training_days": int(len(train)),
        "testing_days": int(len(test)),
        "forecast_days": FORECAST_DAYS,
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_percent": (
            float(mape) if mape is not None else None
        )
    }

    print(f"{currency} forecasting completed!")


# ==========================================
# 12. SAVE ALL RESULTS
# ==========================================

if all_forecasts:

    forecast_df = pd.concat(
        all_forecasts, ignore_index=True
    )

    forecast_df.to_csv(
        os.path.join(
            OUTPUT_DIR, "demand_forecast_30days.csv"
        ),
        index=False
    )

if all_test_predictions:

    test_df = pd.concat(
        all_test_predictions, ignore_index=True
    )

    test_df.to_csv(
        os.path.join(
            OUTPUT_DIR, "forecast_test_predictions.csv"
        ),
        index=False
    )

with open(
    os.path.join(OUTPUT_DIR, "forecast_metrics.json"),
    "w"
) as file:

    json.dump(all_metrics, file, indent=4)


# ==========================================
# 13. FINAL REPORT
# ==========================================

print("\n" + "=" * 50)
print("DEMAND FORECASTING COMPLETED")
print("=" * 50)

print("\nEvaluation Results:")

for currency, metrics in all_metrics.items():

    print(f"\n{currency}")

    for metric, value in metrics.items():

        print(f"{metric}: {value}")

print("\nGenerated files:")
print("1. demand_forecast_30days.csv")
print("2. forecast_test_predictions.csv")
print("3. forecast_metrics.json")
print("4. prophet_INR.joblib")
print("5. prophet_USD.joblib")

print("\nAll forecasting tasks completed!")