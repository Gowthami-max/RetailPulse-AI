import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# SETTINGS
torch.manual_seed(42)
np.random.seed(42)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE, "data", "processed", "orders.csv"
)

OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs", "forecasting"
)

os.makedirs(OUTPUT, exist_ok=True)

LOOKBACK = 12
HORIZON = 8
FOLDS = 5
EPOCHS = 60
BATCH_SIZE = 16


# LOAD DATA
print("\nLoading USD sales data...")

df = pd.read_csv(DATA_FILE)

df["order_datetime"] = pd.to_datetime(
    df["order_datetime"], errors="coerce"
)

df["order_total"] = pd.to_numeric(
    df["order_total"], errors="coerce"
)

df["currency"] = df["currency"].astype(str).str.upper()

df["order_status"] = (
    df["order_status"].astype(str).str.lower()
)

df = df[
    (df["currency"] == "USD") &
    (df["order_status"] == "delivered")
].dropna(subset=["order_datetime", "order_total"])

weekly = (
    df.set_index("order_datetime")
    .resample("W-SUN")["order_total"]
    .sum()
)

last_date = df["order_datetime"].max()

if last_date.date() < weekly.index[-1].date():
    weekly = weekly.iloc[:-1]

weekly = weekly.asfreq("W-SUN").fillna(0)

values = weekly.values.astype(np.float32)

print("Total weeks:", len(values))


# LSTM MODEL
class LSTMModel(nn.Module):

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


# CREATE TRAINING SEQUENCES
def create_sequences(data):

    X, y = [], []

    for i in range(LOOKBACK, len(data)):

        X.append(data[i-LOOKBACK:i])
        y.append(data[i])

    return (
        np.array(X, dtype=np.float32).reshape(-1, LOOKBACK, 1),
        np.array(y, dtype=np.float32).reshape(-1, 1)
    )


# TRAIN MODEL
def train_model(train_values):

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        train_values.reshape(-1, 1)
    )

    X, y = create_sequences(scaled)

    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = LSTMModel()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
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


# RECURSIVE FORECAST
def forecast(model, scaler, history, dates):

    model.eval()

    scaled_history = scaler.transform(
        history.reshape(-1, 1)
    ).flatten().tolist()

    predictions = []

    for _ in dates:

        sequence = np.array(
            scaled_history[-LOOKBACK:],
            dtype=np.float32
        ).reshape(1, LOOKBACK, 1)

        tensor = torch.tensor(sequence)

        with torch.no_grad():
            prediction = model(tensor).item()

        scaled_history.append(prediction)

        actual_prediction = scaler.inverse_transform(
            np.array([[prediction]])
        )[0, 0]

        predictions.append(max(0, actual_prediction))

    return predictions


# ROLLING VALIDATION
print("\nStarting LSTM 5-fold validation...")

fold_results = []
all_predictions = []

for fold in range(FOLDS):

    train_end = len(values) - (FOLDS - fold) * HORIZON

    train_values = values[:train_end]

    test_values = values[
        train_end:train_end + HORIZON
    ]

    test_dates = weekly.index[
        train_end:train_end + HORIZON
    ]

    print(f"\nTraining Fold {fold + 1}")
    print("Training weeks:", len(train_values))
    print("Testing weeks:", len(test_values))

    model, scaler = train_model(train_values)

    predictions = forecast(
        model,
        scaler,
        train_values,
        test_dates
    )

    actual = test_values
    predicted = np.array(predictions)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mape = np.mean(
        np.abs(
            (actual - predicted) /
            np.maximum(np.abs(actual), 1e-8)
        )
    ) * 100

    wape = (
        np.sum(np.abs(actual - predicted)) /
        np.sum(np.abs(actual))
    ) * 100

    print("MAE:", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("MAPE:", round(mape, 2), "%")
    print("WAPE:", round(wape, 2), "%")

    fold_results.append({
        "Fold": fold + 1,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "WAPE": wape
    })

    for date, a, p in zip(test_dates, actual, predicted):

        all_predictions.append({
            "Fold": fold + 1,
            "Date": date,
            "Actual": a,
            "Predicted": p
        })


# FINAL METRICS
results = pd.DataFrame(fold_results)

pred_df = pd.DataFrame(all_predictions)

actual = pred_df["Actual"].values
predicted = pred_df["Predicted"].values

pooled_mae = mean_absolute_error(actual, predicted)

pooled_rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

pooled_mape = np.mean(
    np.abs(
        (actual - predicted) /
        np.maximum(np.abs(actual), 1e-8)
    )
) * 100

pooled_wape = (
    np.sum(np.abs(actual - predicted)) /
    np.sum(np.abs(actual))
) * 100

summary = pd.DataFrame([{
    "Model": "LSTM USD",
    "MAE": pooled_mae,
    "RMSE": pooled_rmse,
    "MAPE": pooled_mape,
    "WAPE": pooled_wape
}])


# SAVE
results.to_csv(
    os.path.join(OUTPUT, "lstm_usd_folds.csv"),
    index=False
)

pred_df.to_csv(
    os.path.join(OUTPUT, "lstm_usd_predictions.csv"),
    index=False
)

summary.to_csv(
    os.path.join(OUTPUT, "lstm_usd_summary.csv"),
    index=False
)

print("\n================================")
print("LSTM FORECASTING RESULTS")
print("================================")

print(summary.to_string(index=False))

print("\nZIDIO MAPE TARGET: <= 12%")

if pooled_mape <= 12:
    print("TARGET ACHIEVED")
else:
    print("TARGET NOT YET ACHIEVED")

print("\nResults saved:", OUTPUT)
print("\nLSTM experiment completed!")