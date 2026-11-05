from pathlib import Path
import pandas as pd, uuid
from datetime import datetime
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data"
seed=pd.read_csv(DATA/"live_transactions_seed.csv"); batch=seed.sample(5,replace=True).copy(); now=datetime.now().astimezone()
batch["demo_event_id"]=[str(uuid.uuid4()) for _ in range(len(batch))]; batch["order_datetime"]=now.strftime("%Y-%m-%d %H:%M:%S"); batch["order_date"]=now.date().isoformat(); batch["data_mode"]="SIMULATED_DEMO"; batch["is_live_simulated"]=True
out=DATA/"simulated_live_transactions.csv"; batch.to_csv(out,mode="a" if out.exists() else "w",header=not out.exists(),index=False); print("Appended 5 simulated demo rows:",out)
