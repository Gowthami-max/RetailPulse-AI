import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(BASE, "data", "processed", "orders.csv")

OUTPUT = os.path.join(
    BASE, "models", "outputs", "forecasting"
)

os.makedirs(OUTPUT, exist_ok=True)

LOOKBACK = 30
FORECAST_DAYS = 30
TEST_DAYS = 30
EPOCHS = 60
BATCH_SIZE = 32

CURRENCIES = ["INR", "USD"]


class DailyLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])


def create_sequences(data):

    X, y = [], []

    for i in range(LOOKBACK, len(data)):
        X.append(data[i - LOOKBACK:i])
        y.append(data[i])

    X = np.array(X, dtype=np.float32).reshape(
        -1, LOOKBACK, 1
    )

    y = np.array(y, dtype=np.float32).reshape(-1, 1)

    return X, y


def train_model(values):

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        values.reshape(-1, 1)
    )

    X, y = create_sequences(scaled)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X),
            torch.tensor(y)
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = DailyLSTM()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    criterion = nn.MSELoss()

    model.train()

    for epoch in range(EPOCHS):

        for batch_x, batch_y in loader:

            optimizer.zero_grad()

            predictions = model(batch_x)

            loss = criterion(predictions, batch_y)

            loss.backward()

            optimizer.step()

    return model, scaler


def recursive_forecast(model, scaler, history, days):

    model.eval()

    scaled_history = scaler.transform(
        history.reshape(-1, 1)
    ).flatten().tolist()

    predictions = []

    for _ in range(days):

        sequence = np.array(
            scaled_history[-LOOKBACK:],
            dtype=np.float32
        ).reshape(1, LOOKBACK, 1)

        tensor = torch.tensor(sequence)

        with torch.no_grad():
            prediction = model(tensor).item()

        scaled_history.append(prediction)

        original = scaler.inverse_transform(
            np.array([[prediction]])
        )[0, 0]

        predictions.append(max(0.0, float(original)))

    return predictions


def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    nonzero = actual != 0

    mape = None

    if nonzero.any():
        mape = float(
            np.mean(
                np.abs(
                    (actual[nonzero] - predicted[nonzero])
                    / actual[nonzero]
                )
            ) * 100
        )

    denominator = np.sum(np.abs(actual))

    wape = None

    if denominator > 0:
        wape = float(
            np.sum(np.abs(actual - predicted))
            / denominator * 100
        )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_Percent": mape,
        "WAPE_Percent": wape
    }


print("\nLoading orders dataset...")

df = pd.read_csv(DATA_FILE)

df["order_date"] = pd.to_datetime(
    df["order_date"], errors="coerce"
)

df["order_total"] = pd.to_numeric(
    df["order_total"], errors="coerce"
)

df["currency"] = (
    df["currency"].astype(str).str.upper().str.strip()
)

df["order_status"] = (
    df["order_status"].astype(str).str.lower().str.strip()
)

df = df.dropna(
    subset=["order_date", "order_total", "currency"]
)

df = df[
    df["order_status"] == "delivered"
].copy()

all_predictions = []
all_forecasts = []
all_metrics = {}

for currency in CURRENCIES:

    print("\n" + "=" * 50)
    print("DAILY LSTM:", currency)
    print("=" * 50)

    currency_df = df[
        df["currency"] == currency
    ].copy()

    daily = (
        currency_df.groupby("order_date")["order_total"]
        .sum()
        .sort_index()
    )

    daily = daily.asfreq("D", fill_value=0)

    values = daily.values.astype(np.float32)
    dates = daily.index

    print("Historical days:", len(values))

    if len(values) <= LOOKBACK + TEST_DAYS:
        print("Insufficient history. Skipping.")
        continue

    # Hold out the last 30 days
    train_values = values[:-TEST_DAYS]
    test_values = values[-TEST_DAYS:]
    test_dates = dates[-TEST_DAYS:]

    print("Training days:", len(train_values))
    print("Testing days:", len(test_values))

    # Train evaluation model
    model, scaler = train_model(train_values)

    test_predictions = recursive_forecast(
        model,
        scaler,
        train_values,
        TEST_DAYS
    )

    test_predictions = np.array(test_predictions)

    metrics = calculate_metrics(
        test_values,
        test_predictions
    )

    print("\nVALIDATION RESULTS")

    for name, value in metrics.items():
        print(name + ":", value)

    for date, actual, predicted in zip(
        test_dates, test_values, test_predictions
    ):

        all_predictions.append({
            "Date": date,
            "Currency": currency,
            "Actual": float(actual),
            "Predicted": float(predicted)
        })

    # Retrain on complete history for future forecast
    print("\nTraining final model on full history...")

    final_model, final_scaler = train_model(values)

    future_predictions = recursive_forecast(
        final_model,
        final_scaler,
        values,
        FORECAST_DAYS
    )

    future_dates = pd.date_range(
        start=dates[-1] + pd.Timedelta(days=1),
        periods=FORECAST_DAYS,
        freq="D"
    )

    for date, prediction in zip(
        future_dates, future_predictions
    ):

        all_forecasts.append({
            "Date": date,
            "Currency": currency,
            "Predicted_Revenue": prediction
        })

    all_metrics[currency] = {
        "training_days": int(len(train_values)),
        "testing_days": TEST_DAYS,
        "forecast_days": FORECAST_DAYS,
        **metrics
    }


# Save outputs
if all_predictions:

    pd.DataFrame(all_predictions).to_csv(
        os.path.join(OUTPUT, "daily_lstm_test_predictions.csv"),
        index=False
    )

if all_forecasts:

    pd.DataFrame(all_forecasts).to_csv(
        os.path.join(OUTPUT, "daily_lstm_forecast_30days.csv"),
        index=False
    )

with open(
    os.path.join(OUTPUT, "daily_lstm_metrics.json"),
    "w"
) as file:

    json.dump(all_metrics, file, indent=4)


print("\n" + "=" * 50)
print("DAILY LSTM COMPLETED")
print("=" * 50)

print(json.dumps(all_metrics, indent=4))

print("\nGenerated files:")
print("1. daily_lstm_test_predictions.csv")
print("2. daily_lstm_forecast_30days.csv")
print("3. daily_lstm_metrics.json")
