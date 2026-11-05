"""
RetailPulse - USD component-based weekly forecasting experiment.

Forecasts weekly delivered USD order count and average order value separately,
then multiplies the forecasts to estimate weekly revenue.

This is an isolated experiment. It does not modify existing models or outputs.
Uses the same five expanding-window folds (108, 116, 124, 132, 140 train weeks)
and 8-week horizon used in the previous USD experiment.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
ORDERS_PATH = ROOT / "data" / "processed" / "orders.csv"
OUTPUT_DIR = ROOT / "models" / "outputs" / "forecasting"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_TRAIN_SIZES = [108, 116, 124, 132, 140]
HORIZON = 8
LAGS = [1, 2, 3, 4, 8, 13, 26, 52]


def load_weekly_components():
    df = pd.read_csv(ORDERS_PATH)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["currency"] = df["currency"].astype(str).str.upper()
    df["order_status"] = df["order_status"].astype(str).str.lower()
    df["order_total"] = pd.to_numeric(df["order_total"], errors="coerce")

    df = df[
        (df["currency"] == "USD")
        & (df["order_status"] == "delivered")
        & df["order_date"].notna()
        & df["order_total"].notna()
    ].copy()

    df["week"] = (
        df["order_date"].dt.to_period("W-SUN").dt.end_time.dt.normalize()
    )

    weekly = df.groupby("week").agg(
        orders=("order_id", "nunique"),
        revenue=("order_total", "sum"),
    )
    weekly["aov"] = weekly["revenue"] / weekly["orders"]
    weekly = weekly.asfreq("W-SUN")

    # Keep the same complete-week convention as the previous experiments:
    # exclude the current/incomplete final week.
    last_date = df["order_date"].max().normalize()
    last_complete_sunday = (
        last_date - pd.Timedelta(days=(last_date.weekday() + 1) % 7 or 7)
    )
    weekly = weekly.loc[weekly.index <= last_complete_sunday]

    weekly[["orders", "revenue", "aov"]] = weekly[
        ["orders", "revenue", "aov"]
    ].fillna(0.0)

    return weekly


def make_features(history, target_date):
    row = {}
    for lag in LAGS:
        row[f"lag_{lag}"] = float(history.iloc[-lag]) if len(history) >= lag else np.nan

    for window in [4, 8, 13, 26]:
        vals = history.iloc[-window:]
        row[f"mean_{window}"] = float(vals.mean()) if len(vals) else np.nan
        row[f"std_{window}"] = float(vals.std(ddof=0)) if len(vals) else 0.0

    row["month"] = target_date.month
    row["quarter"] = target_date.quarter
    row["week_of_year"] = int(target_date.isocalendar().week)
    row["year"] = target_date.year
    return row


def recursive_component_forecast(train_values, train_dates, test_dates, seed):
    history = pd.Series(train_values, index=train_dates, dtype=float)
    predictions = []

    for date in test_dates:
        features = pd.DataFrame([make_features(history, date)])
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=seed,
            n_jobs=-1,
        )

        # Train one-step-ahead examples from the available training history.
        X_rows, y_rows = [], []
        for pos in range(max(LAGS), len(history)):
            hist_slice = history.iloc[:pos]
            X_rows.append(make_features(hist_slice, history.index[pos]))
            y_rows.append(float(history.iloc[pos]))

        X_train = pd.DataFrame(X_rows).replace([np.inf, -np.inf], np.nan)
        X_train = X_train.fillna(X_train.median()).fillna(0.0)
        X_pred = features.reindex(columns=X_train.columns)
        X_pred = X_pred.replace([np.inf, -np.inf], np.nan)
        X_pred = X_pred.fillna(X_train.median()).fillna(0.0)

        model.fit(X_train, np.asarray(y_rows, dtype=float))
        prediction = float(model.predict(X_pred)[0])
        prediction = max(0.0, prediction)
        predictions.append(prediction)
        history.loc[date] = prediction

    return np.asarray(predictions, dtype=float)


def safe_mape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.abs(actual) > 1e-9
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def safe_wape(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.sum(np.abs(actual))
    if denominator <= 1e-9:
        return np.nan
    return float(np.sum(np.abs(actual - predicted)) / denominator * 100)


def main():
    weekly = load_weekly_components()
    print("\n==============================================")
    print(" RETAILPULSE - USD COMPONENT FORECAST EXPERIMENT")
    print("==============================================")
    print(f"Complete USD weeks available: {len(weekly)}")

    fold_rows = []
    prediction_rows = []

    for fold_no, train_size in enumerate(FOLD_TRAIN_SIZES, start=1):
        if train_size + HORIZON > len(weekly):
            print(f"Skipping fold {fold_no}: insufficient weeks.")
            continue

        train = weekly.iloc[:train_size]
        test = weekly.iloc[train_size:train_size + HORIZON]
        test_dates = test.index

        print(f"\nFOLD {fold_no}/5 | Train weeks: {len(train)}")
        print(f"Testing: {test_dates.min().date()} to {test_dates.max().date()}")

        order_pred = recursive_component_forecast(
            train["orders"], train.index, test_dates, seed=101
        )
        aov_pred = recursive_component_forecast(
            train["aov"], train.index, test_dates, seed=202
        )

        revenue_pred = order_pred * aov_pred
        actual_revenue = test["revenue"].to_numpy(dtype=float)

        metrics = {
            "MAE": mean_absolute_error(actual_revenue, revenue_pred),
            "RMSE": np.sqrt(mean_squared_error(actual_revenue, revenue_pred)),
            "MAPE_Percent": safe_mape(actual_revenue, revenue_pred),
            "WAPE_Percent": safe_wape(actual_revenue, revenue_pred),
        }

        print(
            f"Component RF: MAPE={metrics['MAPE_Percent']:.2f}% | "
            f"WAPE={metrics['WAPE_Percent']:.2f}%"
        )

        fold_rows.append({
            "Currency": "USD",
            "Fold": fold_no,
            "Train_Weeks": train_size,
            "Test_Start": test_dates.min().date(),
            "Test_End": test_dates.max().date(),
            "Model": "RF_OrderCount_x_AOV",
            **{k: round(float(v), 4) for k, v in metrics.items()},
        })

        for date, actual_orders, actual_aov, actual_rev, pred_orders, pred_aov, pred_rev in zip(
            test_dates,
            test["orders"].to_numpy(),
            test["aov"].to_numpy(),
            actual_revenue,
            order_pred,
            aov_pred,
            revenue_pred,
        ):
            prediction_rows.append({
                "week": date.date(),
                "fold": fold_no,
                "actual_orders": actual_orders,
                "predicted_orders": pred_orders,
                "actual_aov": actual_aov,
                "predicted_aov": pred_aov,
                "actual_revenue": actual_rev,
                "predicted_revenue": pred_rev,
            })

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)

    if predictions.empty:
        raise RuntimeError("No folds were evaluated. Check weekly data length.")

    actual = predictions["actual_revenue"].to_numpy()
    predicted = predictions["predicted_revenue"].to_numpy()

    summary = pd.DataFrame([{
        "Currency": "USD",
        "Model": "RF_OrderCount_x_AOV",
        "Folds": int(folds["Fold"].nunique()),
        "Forecast_Weeks": len(predictions),
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "MAPE_Percent": safe_mape(actual, predicted),
        "WAPE_Percent": safe_wape(actual, predicted),
        "MAPE_Target_12_Percent": (
            "Met" if safe_mape(actual, predicted) <= 12 else "Not Met"
        ),
    }])

    summary_path = OUTPUT_DIR / "usd_component_forecasting_summary.csv"
    folds_path = OUTPUT_DIR / "usd_component_forecasting_folds.csv"
    preds_path = OUTPUT_DIR / "usd_component_forecasting_predictions.csv"

    summary.to_csv(summary_path, index=False)
    folds.to_csv(folds_path, index=False)
    predictions.to_csv(preds_path, index=False)

    print("\n==============================================")
    print(" COMPONENT FORECASTING SUMMARY")
    print("==============================================")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(summary_path)
    print(folds_path)
    print(preds_path)
    print("\nThis is an experimental candidate, not yet approved for deployment.")


if __name__ == "__main__":
    main()
