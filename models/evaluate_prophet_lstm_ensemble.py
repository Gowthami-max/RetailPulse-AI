import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "outputs", "forecasting")

prophet_path = os.path.join(
    OUTPUT_DIR, "weekly_rolling_backtest_predictions.csv"
)
lstm_path = os.path.join(
    OUTPUT_DIR, "lstm_usd_predictions.csv"
)

prophet = pd.read_csv(prophet_path)
lstm = pd.read_csv(lstm_path)

# Select USD Prophet predictions
prophet = prophet[
    (prophet["Currency"] == "USD") &
    (prophet["Model"] == "Prophet")
].copy()

prophet["Date"] = pd.to_datetime(prophet["Date"])
lstm["Date"] = pd.to_datetime(lstm["Date"])

# Match predictions by fold and date
merged = prophet.merge(
    lstm,
    on=["Fold", "Date"],
    suffixes=("_prophet", "_lstm"),
    how="inner"
)

if len(merged) != 40:
    raise ValueError(
        f"Expected 40 matched predictions, found {len(merged)}"
    )

# Verify actual values
if not np.allclose(
    merged["Actual_prophet"],
    merged["Actual_lstm"]
):
    raise ValueError("Actual values do not match!")

actual = merged["Actual_prophet"].to_numpy()

# Equal-weight ensemble
merged["Ensemble_Predicted"] = (
    0.5 * merged["Predicted_prophet"] +
    0.5 * merged["Predicted_lstm"]
)

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    nonzero = actual != 0

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    mape = np.mean(
        np.abs((actual[nonzero] - predicted[nonzero]) /
               actual[nonzero])
    ) * 100

    wape = (
        np.sum(np.abs(actual - predicted)) /
        np.sum(np.abs(actual))
    ) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_Percent": mape,
        "WAPE_Percent": wape
    }

# Compare all three models
models = {
    "Prophet": merged["Predicted_prophet"],
    "LSTM": merged["Predicted_lstm"],
    "Prophet_LSTM_Ensemble": merged["Ensemble_Predicted"]
}

results = []

for name, predictions in models.items():
    metrics = calculate_metrics(actual, predictions)

    results.append({
        "Currency": "USD",
        "Model": name,
        **metrics,
        "MAPE_Target_12_Percent": (
            "Met" if metrics["MAPE_Percent"] <= 12 else "Not Met"
        )
    })

summary = pd.DataFrame(results)

# Save predictions and evaluation
merged.to_csv(
    os.path.join(OUTPUT_DIR, "prophet_lstm_ensemble_predictions.csv"),
    index=False
)

summary.to_csv(
    os.path.join(OUTPUT_DIR, "prophet_lstm_ensemble_summary.csv"),
    index=False
)

print("\n========== PROPHET + LSTM ENSEMBLE ==========")
print(summary.round(4).to_string(index=False))

# Fold-wise ensemble evaluation
fold_results = []

for fold, group in merged.groupby("Fold"):
    metrics = calculate_metrics(
        group["Actual_prophet"],
        group["Ensemble_Predicted"]
    )

    fold_results.append({
        "Fold": fold,
        **metrics
    })

fold_summary = pd.DataFrame(fold_results)

fold_summary.to_csv(
    os.path.join(OUTPUT_DIR, "prophet_lstm_ensemble_fold_metrics.csv"),
    index=False
)

print("\n========== FOLD-WISE ENSEMBLE RESULTS ==========")
print(fold_summary.round(4).to_string(index=False))

print("\nEnsemble evaluation completed.")
print("Three output CSV files saved.")
