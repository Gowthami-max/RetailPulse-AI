import os
import numpy as np
import pandas as pd

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(base, "models", "outputs", "forecasting")

df = pd.read_csv(
    os.path.join(out, "prophet_lstm_ensemble_predictions.csv")
)

def mape(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100

# Earlier folds for weight selection
train = df[df["Fold"].isin([1, 2, 3])]

# Later folds held out
test = df[df["Fold"].isin([4, 5])]

results = []

for weight in np.arange(0, 1.01, 0.05):
    prediction = (
        weight * train["Predicted_prophet"] +
        (1 - weight) * train["Predicted_lstm"]
    )

    results.append({
        "Prophet_Weight": weight,
        "LSTM_Weight": 1 - weight,
        "Validation_MAPE": mape(
            train["Actual_prophet"], prediction
        )
    })

weights = pd.DataFrame(results)
best = weights.loc[weights["Validation_MAPE"].idxmin()]

prophet_weight = best["Prophet_Weight"]
lstm_weight = best["LSTM_Weight"]

test_prediction = (
    prophet_weight * test["Predicted_prophet"] +
    lstm_weight * test["Predicted_lstm"]
)

print("\n========== CHRONOLOGICAL VALIDATION ==========")
print("Selected Prophet weight:", round(prophet_weight, 2))
print("Selected LSTM weight:", round(lstm_weight, 2))
print("Training folds: 1, 2, 3")
print("Held-out folds: 4, 5")

print("\nHeld-out MAPE:")
print("Prophet:", round(mape(
    test["Actual_prophet"], test["Predicted_prophet"]
), 4))

print("LSTM:", round(mape(
    test["Actual_lstm"], test["Predicted_lstm"]
), 4))

print("Ensemble:", round(mape(
    test["Actual_prophet"], test_prediction
), 4))

weights.to_csv(
    os.path.join(out, "chronological_weight_selection.csv"),
    index=False
)

print("\nWeight selection completed.")
