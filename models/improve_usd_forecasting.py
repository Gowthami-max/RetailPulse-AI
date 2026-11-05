import os
import warnings
import numpy as np
import pandas as pd

from prophet import Prophet
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# =========================================================
# 1. PATHS AND CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "orders.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "outputs", "forecasting")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CURRENCY = "USD"
N_FOLDS = 5
TEST_WEEKS = 8
MIN_TRAIN_WEEKS = 52
LAGS = [1, 2, 4, 8, 13, 26, 52]


# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

def create_features(date, history):
    features = {
        "week": int(date.isocalendar().week),
        "month": int(date.month),
        "quarter": int(date.quarter),
        "year": int(date.year),
    }

    for lag in LAGS:
        features[f"lag_{lag}"] = float(history[-lag])

    features["rolling_mean_4"] = float(np.mean(history[-4:]))
    features["rolling_mean_8"] = float(np.mean(history[-8:]))
    features["rolling_mean_13"] = float(np.mean(history[-13:]))
    features["rolling_std_4"] = float(np.std(history[-4:]))
    features["rolling_std_8"] = float(np.std(history[-8:]))

    return features


# =========================================================
# 3. METRICS
# =========================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    nonzero = actual != 0
    mape = (
        np.mean(np.abs((actual[nonzero] - predicted[nonzero]) /
                       actual[nonzero])) * 100
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


# =========================================================
# 4. LOAD WEEKLY USD SERIES
# =========================================================

def load_usd_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {"order_date", "order_total", "currency", "order_status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"orders.csv is missing columns: {sorted(missing)}")

    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["order_total"] = pd.to_numeric(df["order_total"], errors="coerce")
    df["currency"] = df["currency"].astype(str).str.strip().str.upper()
    df["order_status"] = (
        df["order_status"].astype(str).str.strip().str.lower()
    )

    df = df.dropna(subset=["order_date", "order_total"])
    df = df[
        (df["currency"] == CURRENCY) &
        (df["order_status"] == "delivered")
    ].copy()

    if df.empty:
        raise ValueError("No delivered USD orders were found in orders.csv.")

    weekly = (
        df.set_index("order_date")["order_total"]
        .resample("W-SUN")
        .sum()
    )

    # Match the complete-week handling used by the existing weekly pipeline.
    max_order_date = df["order_date"].max()
    last_complete_sunday = (
        max_order_date -
        pd.Timedelta(days=max_order_date.weekday() + 1)
    ).normalize()

    weekly = weekly[weekly.index <= last_complete_sunday].astype(float)

    if weekly.isna().any():
        raise ValueError("Weekly USD series contains missing values.")

    return weekly


# =========================================================
# 5. MODEL FACTORIES
# =========================================================

def get_model_factories():
    # A fresh estimator is created for each fold.
    return {
        "RF_Baseline": lambda: RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "RF_Regularized": lambda: RandomForestRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=4,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        ),
        "RF_Deep": lambda: RandomForestRegressor(
            n_estimators=500,
            max_depth=15,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra_Trees": lambda: ExtraTreesRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Hist_Gradient_Boosting": lambda: HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


# =========================================================
# 6. RECURSIVE MACHINE-LEARNING FORECAST
# =========================================================

def train_ml_model(train, test_dates, model):
    rows = []

    for i in range(MIN_TRAIN_WEEKS, len(train)):
        history = train.iloc[:i].tolist()
        rows.append({
            **create_features(train.index[i], history),
            "target": float(train.iloc[i]),
        })

    training_df = pd.DataFrame(rows)
    if training_df.empty:
        raise ValueError("Not enough training rows to build lag features.")

    X_train = training_df.drop(columns=["target"])
    y_train = training_df["target"]
    model.fit(X_train, y_train)

    history = train.tolist()
    predictions = []

    for date in test_dates:
        X_next = pd.DataFrame([create_features(date, history)])
        prediction = max(0.0, float(model.predict(X_next)[0]))
        predictions.append(prediction)
        history.append(prediction)

    return predictions


# =========================================================
# 7. PROPHET FORECAST
# =========================================================

def train_prophet(train, test_dates):
    prophet_train = pd.DataFrame({
        "ds": train.index,
        "y": train.values,
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        interval_width=0.95,
    )
    model.fit(prophet_train)

    future = pd.DataFrame({"ds": test_dates})
    forecast = model.predict(future)
    return forecast["yhat"].clip(lower=0).tolist()


# =========================================================
# 8. MAIN ROLLING-WINDOW EXPERIMENT
# =========================================================

def main():
    print("\n==============================================")
    print(" RETAILPULSE - USD FORECASTING EXPERIMENT")
    print("==============================================")
    print(f"Rolling folds: {N_FOLDS}")
    print(f"Forecast horizon per fold: {TEST_WEEKS} weeks")
    print("Models: Seasonal Naive, Random Forest variants,")
    print("        Extra Trees, Hist Gradient Boosting, Prophet")

    weekly = load_usd_data()
    total_weeks = len(weekly)
    required_weeks = MIN_TRAIN_WEEKS + N_FOLDS * TEST_WEEKS

    print(f"\nComplete USD weeks available: {total_weeks}")

    if total_weeks < required_weeks:
        raise ValueError(
            f"Need at least {required_weeks} complete weeks; "
            f"found {total_weeks}."
        )

    first_train_size = total_weeks - N_FOLDS * TEST_WEEKS
    if first_train_size < MIN_TRAIN_WEEKS:
        raise ValueError("Initial training window is shorter than 52 weeks.")

    fold_results = []
    prediction_rows = []
    model_factories = get_model_factories()

    for fold in range(1, N_FOLDS + 1):
        train_end = first_train_size + (fold - 1) * TEST_WEEKS
        test_end = train_end + TEST_WEEKS

        train = weekly.iloc[:train_end]
        test = weekly.iloc[train_end:test_end]
        test_dates = test.index
        actual = test.values.astype(float)

        print("\n----------------------------------------------")
        print(f"FOLD {fold}/{N_FOLDS}")
        print(f"Training weeks: {len(train)}")
        print(
            f"Testing period: {test_dates.min().date()} "
            f"to {test_dates.max().date()}"
        )

        model_predictions = {}

        # Seasonal Naive: same week one year earlier.
        history = train.tolist()
        naive_predictions = []
        for _date in test_dates:
            prediction = float(history[-52])
            naive_predictions.append(prediction)
            history.append(prediction)
        model_predictions["Seasonal Naive"] = naive_predictions

        # Candidate machine-learning models.
        for model_name, factory in model_factories.items():
            print(f"Training {model_name}...")
            model_predictions[model_name] = train_ml_model(
                train, test_dates, factory()
            )

        # Prophet.
        print("Training Prophet...")
        model_predictions["Prophet"] = train_prophet(train, test_dates)

        for model_name, predictions in model_predictions.items():
            metrics = calculate_metrics(actual, predictions)

            fold_results.append({
                "Currency": CURRENCY,
                "Fold": fold,
                "Train_Weeks": len(train),
                "Test_Start": test_dates.min().date().isoformat(),
                "Test_End": test_dates.max().date().isoformat(),
                "Model": model_name,
                **metrics,
            })

            print(
                f"{model_name}: MAPE={metrics['MAPE_Percent']}%, "
                f"WAPE={metrics['WAPE_Percent']}%"
            )

            for date, actual_value, predicted_value in zip(
                test_dates, actual, predictions
            ):
                prediction_rows.append({
                    "Currency": CURRENCY,
                    "Fold": fold,
                    "Date": date.date().isoformat(),
                    "Model": model_name,
                    "Actual": float(actual_value),
                    "Predicted": float(predicted_value),
                })

    # =====================================================
    # 9. AGGREGATE OUT-OF-SAMPLE METRICS
    # =====================================================

    predictions_df = pd.DataFrame(prediction_rows)
    fold_df = pd.DataFrame(fold_results)

    summary_rows = []
    for model_name, group in predictions_df.groupby("Model"):
        metrics = calculate_metrics(group["Actual"], group["Predicted"])
        summary_rows.append({
            "Currency": CURRENCY,
            "Model": model_name,
            "Folds": int(group["Fold"].nunique()),
            "Forecast_Weeks": int(len(group)),
            **metrics,
            "MAPE_Target_12_Percent": (
                "Met" if metrics["MAPE_Percent"] <= 12 else "Not Met"
            ),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["MAPE_Percent", "WAPE_Percent"]
    )

    # =====================================================
    # 10. SAVE NEW FILES ONLY
    # =====================================================

    summary_path = os.path.join(
        OUTPUT_DIR, "usd_forecasting_improvement_summary.csv"
    )
    folds_path = os.path.join(
        OUTPUT_DIR, "usd_forecasting_improvement_folds.csv"
    )
    predictions_path = os.path.join(
        OUTPUT_DIR, "usd_forecasting_improvement_predictions.csv"
    )

    summary_df.to_csv(summary_path, index=False)
    fold_df.to_csv(folds_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    # =====================================================
    # 11. FINAL REPORT
    # =====================================================

    print("\n==============================================")
    print(" USD FORECASTING EXPERIMENT SUMMARY")
    print("==============================================")
    print(summary_df.to_string(index=False))

    best = summary_df.iloc[0]
    print(f"\nBest candidate by pooled MAPE: {best['Model']}")
    print(f"Pooled MAPE: {best['MAPE_Percent']}%")
    print(f"ZIDIO 12% target: {best['MAPE_Target_12_Percent']}")

    print("\nSaved files:")
    print(summary_path)
    print(folds_path)
    print(predictions_path)

    print(
        "\nExperiment completed. Treat the best model as a candidate only; "
        "confirm it on a separate untouched time period before deployment."
    )


if __name__ == "__main__":
    main()
