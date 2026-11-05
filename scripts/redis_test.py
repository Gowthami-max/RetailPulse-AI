import redis
import json
from datetime import datetime

r = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

try:
    r.ping()
    print("REDIS CONNECTED SUCCESSFULLY")

    dashboard_data = {
        "total_tables": 18,
        "database": "retailpulse_db",
        "status": "Connected",
        "timestamp": str(datetime.now())
    }

    r.setex(
        "retailpulse:dashboard:status",
        300,
        json.dumps(dashboard_data)
    )

    cached_data = r.get("retailpulse:dashboard:status")

    print("\nCACHED DASHBOARD DATA:")
    print(json.dumps(json.loads(cached_data), indent=4))

    print("\nREDIS CACHE TEST SUCCESSFUL")

except Exception as e:
    print("REDIS ERROR:", e)