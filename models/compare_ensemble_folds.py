import os
import pandas as pd

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out = os.path.join(base, "models", "outputs", "forecasting")

prophet = pd.read_csv(
    os.path.join(out, "weekly_rolling_backtest_predictions.csv")
)

lstm = pd.read_csv(
    os.path.join(out, "lstm_usd_predictions.csv")
)

prophet = prophet[
    (prophet["Currency"] == "USD") &
    (prophet["Model"] == "Prophet")
].copy()

merged = prophet.merge(
    lstm,
    on=["Fold", "Date"],
    suffixes=("_prophet", "_lstm")
)

rows = []

for fold, group in merged.groupby("Fold"):

    actual = group["Actual_prophet"]

    for model, prediction in [
        ("Prophet", group["Predicted_prophet"]),
        ("LSTM", group["Predicted_lstm"]),
        ("90% LSTM + 10% Prophet",
         0.1 * group["Predicted_prophet"] +
         0.9 * group["Predicted_lstm"])
    ]:

        mape = (
            ((actual - prediction).abs() / actual.abs()).mean()
            * 100
        )

        rows.append({
            "Fold": fold,
            "Model": model,
            "MAPE": round(mape, 4)
        })

result = pd.DataFrame(rows)

print("\n========== FOLD-WISE MODEL COMPARISON ==========")
print(result.to_string(index=False))

result.to_csv(
    os.path.join(out, "fold_wise_model_comparison.csv"),
    index=False
)

print("\nSaved fold_wise_model_comparison.csv")
