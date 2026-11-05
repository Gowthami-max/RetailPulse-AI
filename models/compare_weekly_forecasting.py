import os
import warnings
import numpy as np
import pandas as pd

from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "orders.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR, "models", "outputs", "forecasting"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

TEST_WEEKS = 8
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

    actual = np.array(actual)
    predicted = np.array(predicted)

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    nonzero = actual != 0

    mape = (
        np.mean(
            np.abs(
                (actual[nonzero] - predicted[nonzero])
                / actual[nonzero]
            )
        ) * 100
        if nonzero.any()
        else np.nan
    )

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


def train_random_forest(train, test_dates):

    rows = []

    for i in range(52, len(train)):

        history = train.iloc[:i].tolist()
        date = train.index[i]

        rows.append({
            **create_features(date, history),
            "target": train.iloc[i]
        })

    training_df = pd.DataFrame(rows)

    X_train = training_df.drop(columns=["target"])
    y_train = training_df["target"]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    history = train.tolist()
    predictions = []

    for date in test_dates:

        features = create_features(date, history)

        X = pd.DataFrame([features])

        prediction = max(0, model.predict(X)[0])

        predictions.append(prediction)

        history.append(prediction)

    return model, predictions


def train_prophet(train, test_dates):

    prophet_train = pd.DataFrame({
        "ds": train.index,
        "y": train.values
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95
    )

    model.fit(prophet_train)

    future = pd.DataFrame({"ds": test_dates})

    forecast = model.predict(future)

    predictions = forecast["yhat"].clip(lower=0).tolist()

    return model, predictions


def main():

    print("\n======================================")
    print(" RETAILPULSE WEEKLY FORECASTING")
    print("======================================")

    df = pd.read_csv(DATA_PATH)

    df["order_date"] = pd.to_datetime(df["order_date"])

    df["order_status"] = (
        df["order_status"].astype(str).str.strip().str.lower()
    )

    df = df[df["order_status"] == "delivered"].copy()

    results = []
    prediction_results = []

    for currency in ["INR", "USD"]:

        print(f"\n========== {currency} ==========")

        currency_df = df[df["currency"] == currency]

        weekly = (
            currency_df
            .set_index("order_date")["order_total"]
            .resample("W-SUN")
            .sum()
        )

        # Exclude incomplete final week
        last_complete_sunday = (
            df["order_date"].max()
            - pd.Timedelta(days=df["order_date"].max().weekday() + 1)
        ).normalize()

        weekly = weekly[weekly.index <= last_complete_sunday]

        if len(weekly) <= 52 + TEST_WEEKS:
            print("Not enough weekly history.")
            continue

        train = weekly.iloc[:-TEST_WEEKS]
        test = weekly.iloc[-TEST_WEEKS:]

        test_dates = test.index
        actual = test.values

        print("Total complete weeks:", len(weekly))
        print("Training weeks:", len(train))
        print("Testing weeks:", len(test))

        # 1. Seasonal Naive: same week last year
        naive_predictions = []

        history = train.tolist()

        for date in test_dates:

            prediction = history[-52]

            naive_predictions.append(prediction)

            history.append(prediction)

        # 2. Random Forest
        print("\nTraining Random Forest...")

        rf_model, rf_predictions = train_random_forest(
            train, test_dates
        )

        # 3. Prophet
        print("Training Prophet...")

        prophet_model, prophet_predictions = train_prophet(
            train, test_dates
        )

        model_predictions = {
            "Seasonal Naive": naive_predictions,
            "Random Forest": rf_predictions,
            "Prophet": prophet_predictions
        }

        for model_name, predictions in model_predictions.items():

            metrics = calculate_metrics(actual, predictions)

            results.append({
                "Currency": currency,
                "Model": model_name,
                **metrics
            })

            print(f"\n{model_name}: {metrics}")

            for date, actual_value, predicted_value in zip(
                test_dates, actual, predictions
            ):

                prediction_results.append({
                    "Currency": currency,
                    "Date": date,
                    "Model": model_name,
                    "Actual": actual_value,
                    "Predicted": predicted_value
                })

    results_df = pd.DataFrame(results)

    predictions_df = pd.DataFrame(prediction_results)

    results_df.to_csv(
        os.path.join(OUTPUT_DIR, "weekly_model_comparison.csv"),
        index=False
    )

    predictions_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "weekly_model_comparison_predictions.csv"
        ),
        index=False
    )

    print("\n======================================")
    print(" FINAL WEEKLY MODEL COMPARISON")
    print("======================================")

    print(
        results_df.sort_values(
            ["Currency", "WAPE_Percent"]
        ).to_string(index=False)
    )

    print("\nWeekly forecasting comparison completed!")


if __name__ == "__main__":
    main()