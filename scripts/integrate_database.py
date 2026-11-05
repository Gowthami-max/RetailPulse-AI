from pathlib import Path

app_file = Path("frontend/app.py")
source = app_file.read_text(encoding="utf-8")

start_marker = "APP_DIR = Path(__file__).resolve().parent"
end_marker = "D = get_all()"

start = source.find(start_marker)
end = source.find(end_marker)

if start == -1 or end == -1 or end < start:
    raise RuntimeError("Could not locate the expected data-loading section.")

end += len(end_marker)

new_section = '''REQUIRED_FILES = [
    "customers.csv", "products.csv", "orders.csv", "order_items.csv",
    "payments.csv", "shipments.csv", "returns.csv", "inventory_snapshot.csv",
    "campaigns.csv", "transactions_flat.csv", "customer_analytics_rfm.csv",
    "product_performance.csv", "country_sales_summary.csv",
    "daily_sales_summary.csv", "monthly_sales_summary.csv",
    "payment_summary.csv", "live_transactions_seed.csv",
]

from db_loader import load_table

@st.cache_data(ttl=60, show_spinner="Loading RetailPulse database...")
def get_all():
    return {
        Path(filename).stem: load_table(Path(filename).stem)
        for filename in REQUIRED_FILES
    }

D = get_all()'''

source = source[:start] + new_section + source[end:]

old_caption = 'st.sidebar.caption(f"Data directory: `{DATA_DIR.relative_to(PROJECT_ROOT) if DATA_DIR.is_relative_to(PROJECT_ROOT) else DATA_DIR}`")'

if old_caption not in source:
    raise RuntimeError("Could not locate the expected sidebar data-directory caption.")

source = source.replace(
    old_caption,
    'st.sidebar.caption("Data source: PostgreSQL + Redis")'
)

app_file.write_text(source, encoding="utf-8")

print("SUCCESS: Dashboard data-loading section updated.")
print("Existing dashboard pages and filters were preserved.")