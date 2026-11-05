import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "models", "outputs", "forecasting")

df = pd.read_csv(
    os.path.join(OUT, "prophet_lstm_ensemble_predictions.csv")
)

actual = df["Actual_prophet"].to_numpy()
prophet = df["Predicted_prophet"].to_numpy()
lstm = df["Predicted_lstm"].to_numpy()

results = []

for prophet_weight in np.arange(0, 1.01, 0.1):
    lstm_weight = 1 - prophet_weight

    prediction = (
        prophet_weight * prophet +
        lstm_weight * lstm
    )

    mape = np.mean(
        np.abs((actual - prediction) / actual)
    ) * 100

    wape = (
        np.sum(np.abs(actual - prediction)) /
        np.sum(np.abs(actual))
    ) * 100

    results.append({
        "Prophet_Weight": round(prophet_weight, 2),
        "LSTM_Weight": round(lstm_weight, 2),
        "MAPE": round(mape, 4),
        "WAPE": round(wape, 4)
    })

result = pd.DataFrame(results)

print("\n========== ENSEMBLE WEIGHT COMPARISON ==========")
print(result.to_string(index=False))

result.to_csv(
    os.path.join(OUT, "ensemble_weight_comparison.csv"),
    index=False
)

print("\nSaved ensemble_weight_comparison.csv")
