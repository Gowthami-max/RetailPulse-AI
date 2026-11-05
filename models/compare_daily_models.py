import os
import pandas as pd
import numpy as np

folder = "models/outputs/forecasting"

prophet = pd.read_csv(
    os.path.join(folder, "forecast_test_predictions.csv")
)

lstm = pd.read_csv(
    os.path.join(folder, "daily_lstm_test_predictions.csv")
)

prophet["date"] = pd.to_datetime(prophet["date"])
lstm["Date"] = pd.to_datetime(lstm["Date"])

merged = prophet.merge(
    lstm,
    left_on=["date", "currency"],
    right_on=["Date", "Currency"],
    how="inner"
)

print("\n========== MODEL ALIGNMENT ==========")
print("Prophet rows:", len(prophet))
print("LSTM rows:", len(lstm))
print("Matched rows:", len(merged))

for currency in ["INR", "USD"]:

    group = merged[merged["currency"] == currency]

    print("\n==========", currency, "==========")

    actual = group["actual_revenue"].values

    print("Matched days:", len(group))
    print("Zero revenue days:", (actual == 0).sum())
    print("Average revenue:", round(actual.mean(), 2))
    print("Minimum revenue:", round(actual.min(), 2))
    print("Maximum revenue:", round(actual.max(), 2))

    if len(group) > 0:

        for name, prediction in [
            ("Prophet", group["predicted_revenue"].values),
            ("LSTM", group["Predicted"].values)
        ]:

            nonzero = actual != 0

            mape = np.mean(
                np.abs(
                    (actual[nonzero] - prediction[nonzero]) /
                    actual[nonzero]
                )
            ) * 100

            wape = (
                np.sum(np.abs(actual - prediction)) /
                np.sum(np.abs(actual))
            ) * 100

            print(
                name,
                "| MAPE:", round(mape, 2),
                "| WAPE:", round(wape, 2)
            )

print("\n========== COMPLETED ==========")
