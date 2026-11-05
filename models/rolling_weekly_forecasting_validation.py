import os
import warnings
import numpy as np
import pandas as pd

from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "orders.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "outputs", "forecasting")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Five expanding-window folds. Each fold forecasts the next eight weeks.
TEST_WEEKS = 8
N_FOLDS = 5
MIN_TRAIN_WEEKS = 52
LAGS = [1, 2, 4, 8, 13, 26, 52]


def create_features(date, history):
    features = {
        "week": date.isocalendar().week,
        "month": date.month,
        "quarter": date.quarter,
    }

    for lag in LAGS:
        features[f"lag_{lag}"] = history[-lag]

    features["rolling_mean_4"] = np.mean(history[-4:])
    features["rolling_mean_8"] = np.mean(history[-8:])
    features["rolling_mean_13"] = np.mean(history[-13:])
    return features


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    nonzero = actual != 0
    mape = (
        np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
        if nonzero.any()
        else np.nan
    )

    denominator = np.sum(np.abs(actual))
    wape = (
        np.sum(np.abs(actual - predicted)) / denominator * 100
        if denominator > 0
        else np.nan
    )

    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE_Percent": round(float(mape), 2) if np.isfinite(mape) else np.nan,
        "WAPE_Percent": round(float(wape), 2) if np.isfinite(wape) else np.nan,
    }


def train_random_forest(train, test_dates):
    rows = []

    for i in range(MIN_TRAIN_WEEKS, len(train)):
        history = train.iloc[:i].tolist()
        date = train.index[i]
        rows.append({
            **create_features(date, history),
            "target": train.iloc[i],
        })

    training_df = pd.DataFrame(rows)
    X_train = training_df.drop(columns=["target"])
    y_train = training_df["target"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    history = train.tolist()
    predictions = []

    for date in test_dates:
        features = create_features(date, history)
        prediction = max(0.0, float(model.predict(pd.DataFrame([features]))[0]))
        predictions.append(prediction)
        history.append(prediction)

    return predictions


def train_prophet(train, test_dates):
    prophet_train = pd.DataFrame({
        "ds": train.index,
        "y": train.values,
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95,
    )
    model.fit(prophet_train)

    future = pd.DataFrame({"ds": test_dates})
    forecast = model.predict(future)
    return forecast["yhat"].clip(lower=0).tolist()


def prepare_weekly_series(df, currency):
    currency_df = df[df["currency"].astype(str).str.upper() == currency].copy()

    weekly = (
        currency_df
        .set_index("order_date")["order_total"]
        .resample("W-SUN")
        .sum()
    )

    # Preserve the same incomplete-final-week exclusion used by the
    # existing weekly forecasting pipeline.
    max_order_date = df["order_date"].max()
    last_complete_sunday = (
        max_order_date - pd.Timedelta(days=max_order_date.weekday() + 1)
    ).normalize()

    return weekly[weekly.index <= last_complete_sunday]


def main():
    print("\n==============================================")
    print(" RETAILPULSE - WEEKLY ROLLING-WINDOW BACKTEST")
    print("==============================================")
    print(f"Folds: {N_FOLDS} | Forecast horizon: {TEST_WEEKS} weeks")
    print(f"Minimum training history: {MIN_TRAIN_WEEKS} weeks")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    required_columns = {"order_date", "order_status", "currency", "order_total"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"orders.csv is missing required columns: {sorted(missing)}")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["order_status"] = df["order_status"].astype(str).str.strip().str.lower()
    df["currency"] = df["currency"].astype(str).str.strip().str.upper()
    df["order_total"] = pd.to_numeric(df["order_total"], errors="coerce")
    df = df.dropna(subset=["order_date", "order_total"])
    df = df[df["order_status"] == "delivered"].copy()

    all_fold_metrics = []
    all_predictions = []

    for currency in ["INR", "USD"]:
        print(f"\n========== {currency} ==========")
        weekly = prepare_weekly_series(df, currency)
        total_weeks = len(weekly)
        required_weeks = MIN_TRAIN_WEEKS + (N_FOLDS * TEST_WEEKS)

        if total_weeks < required_weeks:
            print(
                f"Skipping {currency}: {total_weeks} weeks available; "
                f"at least {required_weeks} required."
            )
            continue

        # Expanding windows: each fold adds eight weeks to the training set.
        first_train_size = total_weeks - (N_FOLDS * TEST_WEEKS)

        if first_train_size < MIN_TRAIN_WEEKS:
            print(f"Skipping {currency}: insufficient initial training history.")
            continue

        for fold_number in range(1, N_FOLDS + 1):
            train_end = first_train_size + ((fold_number - 1) * TEST_WEEKS)
            test_end = train_end + TEST_WEEKS

            train = weekly.iloc[:train_end]
            test = weekly.iloc[train_end:test_end]

            if len(test) != TEST_WEEKS:
                continue

            test_dates = test.index
            actual = test.values.astype(float)

            print(
                f"\nFold {fold_number}/{N_FOLDS} | "
                f"Train: {len(train)} weeks | "
                f"Test: {test.index.min().date()} to {test.index.max().date()}"
            )

            # 1. Seasonal Naive: forecast using the same week one year earlier.
            history = train.tolist()
            naive_predictions = []
            for _date in test_dates:
                prediction = float(history[-52])
                naive_predictions.append(prediction)
                history.append(prediction)

            # 2. Random Forest
            print("  Training Random Forest...")
            rf_predictions = train_random_forest(train, test_dates)

            # 3. Prophet
            print("  Training Prophet...")
            prophet_predictions = train_prophet(train, test_dates)

            model_predictions = {
                "Seasonal Naive": naive_predictions,
                "Random Forest": rf_predictions,
                "Prophet": prophet_predictions,
            }

            for model_name, predictions in model_predictions.items():
                fold_metrics = calculate_metrics(actual, predictions)
                all_fold_metrics.append({
                    "Currency": currency,
                    "Fold": fold_number,
                    "Train_Weeks": len(train),
                    "Test_Start": test_dates.min().date().isoformat(),
                    "Test_End": test_dates.max().date().isoformat(),
                    "Model": model_name,
                    **fold_metrics,
                })

                for date, actual_value, predicted_value in zip(
                    test_dates, actual, predictions
                ):
                    all_predictions.append({
                        "Currency": currency,
                        "Fold": fold_number,
                        "Date": date.date().isoformat(),
                        "Model": model_name,
                        "Actual": float(actual_value),
                        "Predicted": float(predicted_value),
                    })

                print(
                    f"  {model_name}: "
                    f"MAPE={fold_metrics['MAPE_Percent']}%, "
                    f"WAPE={fold_metrics['WAPE_Percent']}%"
                )

    if not all_predictions:
        raise RuntimeError(
            "No backtest predictions were generated. Check the dataset history."
        )

    fold_metrics_df = pd.DataFrame(all_fold_metrics)
    predictions_df = pd.DataFrame(all_predictions)

    # Aggregate metrics across all out-of-sample weeks in all folds.
    # These pooled metrics are more informative than averaging fold percentages.
    aggregate_rows = []
    for (currency, model_name), group in predictions_df.groupby(["Currency", "Model"]):
        metrics = calculate_metrics(group["Actual"], group["Predicted"])
        aggregate_rows.append({
            "Currency": currency,
            "Model": model_name,
            "Folds": int(group["Fold"].nunique()),
            "Forecast_Weeks": int(len(group)),
            **metrics,
            "MAPE_Target_12_Percent": (
                "Met" if metrics["MAPE_Percent"] <= 12 else "Not Met"
            ),
        })

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(
        ["Currency", "WAPE_Percent"]
    )

    fold_metrics_path = os.path.join(
        OUTPUT_DIR, "weekly_rolling_backtest_fold_metrics.csv"
    )
    predictions_path = os.path.join(
        OUTPUT_DIR, "weekly_rolling_backtest_predictions.csv"
    )
    aggregate_path = os.path.join(
        OUTPUT_DIR, "weekly_rolling_backtest_summary.csv"
    )

    # These are new filenames; existing weekly comparison files are untouched.
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)

    print("\n==============================================")
    print(" ROLLING-WINDOW AGGREGATE RESULTS")
    print("==============================================")
    print(aggregate_df.to_string(index=False))

    print("\nSaved new output files:")
    print(f"1. {fold_metrics_path}")
    print(f"2. {predictions_path}")
    print(f"3. {aggregate_path}")
    print("\nRolling-window validation completed.")


if __name__ == "__main__":
    main()
