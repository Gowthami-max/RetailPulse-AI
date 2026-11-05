import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

warnings.filterwarnings("ignore")

# -------------------------------------------------
# PATHS
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "orders.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "models", "outputs", "forecasting"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_DAYS = 30


# -------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------

def create_features(date, history):

    return {
        "day_of_week": date.dayofweek,
        "day_of_month": date.day,
        "month": date.month,
        "day_of_year": date.dayofyear,
        "is_weekend": int(date.dayofweek >= 5),

        "lag_1": history[-1],
        "lag_7": history[-7],
        "lag_14": history[-14],
        "lag_28": history[-28],

        "rolling_mean_7": np.mean(history[-7:]),
        "rolling_mean_14": np.mean(history[-14:])
    }


# -------------------------------------------------
# METRICS
# -------------------------------------------------

def calculate_metrics(actual, predicted):

    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    nonzero = actual != 0

    if nonzero.sum() > 0:
        mape = np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
    else:
        mape = np.nan

    wape = (
        np.sum(np.abs(actual - predicted))
        / np.sum(np.abs(actual)) * 100
        if np.sum(np.abs(actual)) > 0
        else np.nan
    )

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE_Percent": round(mape, 2),
        "WAPE_Percent": round(wape, 2)
    }


# -------------------------------------------------
# RANDOM FOREST MODEL
# -------------------------------------------------

def train_random_forest(train_series, test_dates):

    training_rows = []

    for i in range(28, len(train_series)):

        date = train_series.index[i]
        history = train_series.iloc[:i].tolist()

        features = create_features(date, history)

        training_rows.append({
            **features,
            "target": train_series.iloc[i]
        })

    training_df = pd.DataFrame(training_rows)

    X_train = training_df.drop(columns=["target"])
    y_train = training_df["target"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Recursive prediction:
    # Future predictions are used as future lag values.

    history = train_series.tolist()
    predictions = []

    for date in test_dates:

        features = create_features(date, history)

        X = pd.DataFrame([features])

        prediction = model.predict(X)[0]

        prediction = max(0, prediction)

        predictions.append(prediction)

        history.append(prediction)

    return model, predictions


# -------------------------------------------------
# PROPHET MODEL
# -------------------------------------------------

def train_prophet(train_series, test_dates):

    prophet_train = pd.DataFrame({
        "ds": train_series.index,
        "y": train_series.values
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95
    )

    model.fit(prophet_train)

    future = pd.DataFrame({
        "ds": test_dates
    })

    forecast = model.predict(future)

    predictions = forecast["yhat"].clip(lower=0).tolist()

    return model, predictions


# -------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------

def main():

    print("\n======================================")
    print(" RETAILPULSE MODEL COMPARISON")
    print("======================================\n")

    df = pd.read_csv(DATA_PATH)

    df["order_date"] = pd.to_datetime(df["order_date"])

    df["order_status"] = (
        df["order_status"].astype(str).str.strip().str.lower()
    )

    # Use delivered orders only
    df = df[df["order_status"] == "delivered"].copy()

    all_metrics = []
    all_predictions = []

    for currency in ["INR", "USD"]:

        print(f"\n========== {currency} ==========")

        currency_df = df[
            df["currency"] == currency
        ].copy()

        daily = (
            currency_df
            .groupby("order_date")["order_total"]
            .sum()
            .sort_index()
        )

        # Fill missing calendar dates
        daily = daily.asfreq("D", fill_value=0)

        if len(daily) <= TEST_DAYS + 28:
            print("Not enough data. Skipping.")
            continue

        train = daily.iloc[:-TEST_DAYS]
        test = daily.iloc[-TEST_DAYS:]

        test_dates = test.index
        actual = test.values

        print("Training days:", len(train))
        print("Testing days:", len(test))

        # -----------------------------------------
        # 1. SEASONAL NAIVE BASELINE
        # -----------------------------------------

        naive_predictions = []

        history = train.tolist()

        for date in test_dates:

            prediction = history[-7]

            naive_predictions.append(prediction)

            history.append(prediction)

        # -----------------------------------------
        # 2. RANDOM FOREST
        # -----------------------------------------

        print("\nTraining Random Forest...")

        rf_model, rf_predictions = train_random_forest(
            train, test_dates
        )

        # -----------------------------------------
        # 3. PROPHET
        # -----------------------------------------

        print("Training Prophet...")

        prophet_model, prophet_predictions = train_prophet(
            train, test_dates
        )

        # -----------------------------------------
        # EVALUATION
        # -----------------------------------------

        model_predictions = {
            "Seasonal Naive": naive_predictions,
            "Random Forest": rf_predictions,
            "Prophet": prophet_predictions
        }

        for model_name, predictions in model_predictions.items():

            metrics = calculate_metrics(
                actual, predictions
            )

            all_metrics.append({
                "Currency": currency,
                "Model": model_name,
                **metrics
            })

            print(f"\n{model_name}")
            print(metrics)

            for date, actual_value, predicted_value in zip(
                test_dates, actual, predictions
            ):

                all_predictions.append({
                    "Currency": currency,
                    "Date": date,
                    "Model": model_name,
                    "Actual": actual_value,
                    "Predicted": predicted_value
                })

    # ---------------------------------------------
    # SAVE RESULTS
    # ---------------------------------------------

    metrics_df = pd.DataFrame(all_metrics)

    predictions_df = pd.DataFrame(all_predictions)

    metrics_path = os.path.join(
        OUTPUT_DIR, "model_comparison.csv"
    )

    predictions_path = os.path.join(
        OUTPUT_DIR, "model_comparison_predictions.csv"
    )

    metrics_df.to_csv(metrics_path, index=False)

    predictions_df.to_csv(
        predictions_path, index=False
    )

    print("\n======================================")
    print(" FINAL MODEL COMPARISON")
    print("======================================")

    print(
        metrics_df.sort_values(
            ["Currency", "WAPE_Percent"]
        ).to_string(index=False)
    )

    print("\nFiles saved successfully:")
    print(metrics_path)
    print(predictions_path)

    print("\nComparison completed!")


if __name__ == "__main__":
    main()