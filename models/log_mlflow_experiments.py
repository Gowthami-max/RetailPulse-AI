import os
import pandas as pd
import mlflow

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "outputs", "forecasting"
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("RetailPulse Demand Forecasting")

experiments = {
    "RF Component": "usd_component_forecasting_summary.csv",
    "Direct Random Forest": "direct_usd_summary.csv",
    "LSTM": "lstm_usd_summary.csv"
}

for model_name, filename in experiments.items():

    path = os.path.join(OUTPUT, filename)

    if not os.path.exists(path):
        print("Missing:", filename)
        continue

    df = pd.read_csv(path)

    row = df.iloc[0]

    def get_metric(*names):
        for name in names:
            if name in df.columns and pd.notna(row[name]):
                return float(row[name])
        return None

    metrics = {
        "MAE": get_metric("MAE"),
        "RMSE": get_metric("RMSE"),
        "MAPE": get_metric("MAPE", "MAPE_Percent"),
        "WAPE": get_metric("WAPE", "WAPE_Percent")
    }

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param("model", model_name)
        mlflow.log_param("currency", "USD")
        mlflow.log_param("validation", "5-fold rolling validation")

        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, value)

        mlflow.log_artifact(path)

        print("Logged:", model_name, metrics)

print("\nAll available forecasting experiments logged!")