import os
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ORDERS_FILE = os.path.join(
    BASE_DIR, "data", "processed", "orders.csv"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs", "forecasting"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

HORIZON = 8
N_FOLDS = 5

LAGS = [1, 2, 3, 4, 5, 8, 13, 26, 52]
ROLLING_WINDOWS = [4, 8, 13, 26]


# --------------------------------------------------
# 2. LOAD AND PREPARE USD REVENUE
# --------------------------------------------------

print("\nLoading order data...")

orders = pd.read_csv(ORDERS_FILE)

orders["order_datetime"] = pd.to_datetime(
    orders["order_datetime"], errors="coerce"
)

orders["order_total"] = pd.to_numeric(
    orders["order_total"], errors="coerce"
)

orders["currency"] = orders["currency"].astype(str).str.upper()

orders["order_status"] = (
    orders["order_status"].astype(str).str.strip().str.lower()
)

usd = orders[
    (orders["currency"] == "USD") &
    (orders["order_status"] == "delivered")
].copy()

usd = usd.dropna(subset=["order_datetime", "order_total"])

weekly = (
    usd.set_index("order_datetime")
    .resample("W-SUN")["order_total"]
    .sum()
)

# Remove incomplete final week
last_date = usd["order_datetime"].max()
last_week_end = weekly.index[-1]

if last_date.date() < last_week_end.date():
    weekly = weekly.iloc[:-1]

weekly = weekly.asfreq("W-SUN").fillna(0)

print("Complete weeks:", len(weekly))
print("Date range:", weekly.index.min(), "to", weekly.index.max())


# --------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------

def create_features(history, target_date, position):

    row = {}

    for lag in LAGS:
        row[f"lag_{lag}"] = (
            history.iloc[-lag] if len(history) >= lag else np.nan
        )

    for window in ROLLING_WINDOWS:

        values = history.iloc[-window:]

        row[f"rolling_mean_{window}"] = (
            values.mean() if len(values) >= window else np.nan
        )

        row[f"rolling_std_{window}"] = (
            values.std() if len(values) >= window else np.nan
        )

    row["week_of_year"] = target_date.isocalendar().week
    row["month"] = target_date.month
    row["quarter"] = target_date.quarter

    row["week_sin"] = np.sin(
        2 * np.pi * target_date.isocalendar().week / 52
    )

    row["week_cos"] = np.cos(
        2 * np.pi * target_date.isocalendar().week / 52
    )

    row["trend"] = position

    return row


# --------------------------------------------------
# 4. TRAINING DATA CREATION
# --------------------------------------------------

def prepare_training_data(series):

    records = []

    for i in range(52, len(series)):

        history = series.iloc[:i]
        target_date = series.index[i]

        features = create_features(
            history, target_date, i
        )

        features["target"] = series.iloc[i]

        records.append(features)

    return pd.DataFrame(records).dropna()


# --------------------------------------------------
# 5. RANDOM FOREST MODEL
# --------------------------------------------------

def build_model():

    return RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )


# --------------------------------------------------
# 6. RECURSIVE FORECASTING
# --------------------------------------------------

def forecast_future(train_series, future_dates, model):

    history = train_series.copy()
    predictions = []

    for date in future_dates:

        features = create_features(
            history, date, len(history)
        )

        X = pd.DataFrame([features]).fillna(0)

        prediction = model.predict(X)[0]

        prediction = max(0, prediction)

        predictions.append(prediction)

        history.loc[date] = prediction

    return predictions


# --------------------------------------------------
# 7. FIVE-FOLD ROLLING VALIDATION
# --------------------------------------------------

print("\nStarting 5-fold validation...")

fold_results = []
all_predictions = []

total_weeks = len(weekly)

for fold in range(N_FOLDS):

    train_end = total_weeks - (N_FOLDS - fold) * HORIZON

    train_series = weekly.iloc[:train_end]
    test_series = weekly.iloc[
        train_end:train_end + HORIZON
    ]

    print(f"\nFold {fold + 1}")
    print("Training weeks:", len(train_series))
    print("Testing weeks:", len(test_series))

    training_data = prepare_training_data(train_series)

    X_train = training_data.drop(columns=["target"])
    y_train = training_data["target"]

    model = build_model()
    model.fit(X_train, y_train)

    predictions = forecast_future(
        train_series,
        test_series.index,
        model
    )

    actual = test_series.values
    predicted = np.array(predictions)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mape = np.mean(
        np.abs((actual - predicted) / np.maximum(actual, 1e-8))
    ) * 100

    wape = (
        np.sum(np.abs(actual - predicted)) /
        np.sum(np.abs(actual))
    ) * 100

    fold_results.append({
        "fold": fold + 1,
        "train_weeks": len(train_series),
        "test_weeks": len(test_series),
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "WAPE": wape
    })

    for date, actual_value, predicted_value in zip(
        test_series.index, actual, predicted
    ):

        all_predictions.append({
            "fold": fold + 1,
            "date": date,
            "actual": actual_value,
            "predicted": predicted_value
        })

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"WAPE: {wape:.2f}%")


# --------------------------------------------------
# 8. POOLED METRICS
# --------------------------------------------------

pred_df = pd.DataFrame(all_predictions)

actual = pred_df["actual"].values
predicted = pred_df["predicted"].values

pooled_mae = mean_absolute_error(actual, predicted)

pooled_rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

pooled_mape = np.mean(
    np.abs((actual - predicted) / np.maximum(actual, 1e-8))
) * 100

pooled_wape = (
    np.sum(np.abs(actual - predicted)) /
    np.sum(np.abs(actual))
) * 100


# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

fold_df = pd.DataFrame(fold_results)

summary = pd.DataFrame([{
    "Model": "Direct USD Random Forest",
    "Folds": N_FOLDS,
    "Horizon": HORIZON,
    "Total_Test_Weeks": len(pred_df),
    "MAE": pooled_mae,
    "RMSE": pooled_rmse,
    "MAPE": pooled_mape,
    "WAPE": pooled_wape
}])

fold_df.to_csv(
    os.path.join(OUTPUT_DIR, "direct_usd_folds.csv"),
    index=False
)

pred_df.to_csv(
    os.path.join(OUTPUT_DIR, "direct_usd_predictions.csv"),
    index=False
)

summary.to_csv(
    os.path.join(OUTPUT_DIR, "direct_usd_summary.csv"),
    index=False
)


# --------------------------------------------------
# 10. FINAL RESULTS
# --------------------------------------------------

print("\n" + "=" * 50)
print("DIRECT USD FORECASTING RESULTS")
print("=" * 50)

print(f"Pooled MAE:  {pooled_mae:.2f}")
print(f"Pooled RMSE: {pooled_rmse:.2f}")
print(f"Pooled MAPE: {pooled_mape:.2f}%")
print(f"Pooled WAPE: {pooled_wape:.2f}%")

print("\nZIDIO MAPE TARGET: <= 12%")

if pooled_mape <= 12:
    print("RESULT: TARGET ACHIEVED")
else:
    print("RESULT: TARGET NOT YET ACHIEVED")

print("\nResults saved in:", OUTPUT_DIR)
print("\nExperiment completed!")