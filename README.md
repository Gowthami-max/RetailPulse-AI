# RetailPulse -- AI-Powered Customer Analytics & Demand Forecasting Platform

RetailPulse is a Streamlit-based retail analytics dashboard for
exploring customer, sales, product, payment, shipment, return, campaign,
and inventory data.

> **Dataset notice:** The supplied demonstration dataset contains
> synthetic records generated for development and demonstration. It is
> not real customer or retailer transaction data. Do not present its
> results as actual business or financial performance.

## Features

The dashboard contains 12 workspace pages:

1.  Executive Summary
2.  Sales Trend Analysis
3.  Product Performance
4.  Customer Analytics
5.  Region & Country Sales
6.  Monthly & Seasonal Sales
7.  Customer Behaviour
8.  Revenue & Profit Analysis
9.  Inventory Risk
10. Orders & Transactions
11. Advanced Analytics
12. Interactive Filters

Global filters include order date range, country, exact-match Customer
ID, sales channel, and currency view (where available). Customer-linked
records are filtered through the selected customer and related orders.
Inventory is product-level data and is not associated with an individual
customer. Do not combine different currencies without a documented
exchange-rate conversion.

## Technology stack

-   Python 3.11
-   Streamlit
-   Pandas
-   NumPy
-   Plotly

## Project structure

``` text
RetailPulse_AI/
├── backend/
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   └── components/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── utils/
├── database/
├── assets/
├── requirements.txt
└── README.md
```

The exact files in `models/`, `utils/`, and `database/` depend on what
has actually been implemented. Empty folders are placeholders, not
completed features.

## Dataset setup

Place the supplied processed CSV files in `data/processed/`. Common
files include:

-   `customers.csv`
-   `products.csv`
-   `orders.csv`
-   `order_items.csv`
-   `payments.csv`
-   `shipments.csv`
-   `returns.csv`
-   `inventory_snapshot.csv`
-   `campaigns.csv`
-   `transactions_flat.csv`
-   `customer_analytics_rfm.csv`
-   `product_performance.csv`
-   `country_sales_summary.csv`
-   `daily_sales_summary.csv`
-   `monthly_sales_summary.csv`
-   `payment_summary.csv`
-   `live_transactions_seed.csv`

Keep original/raw source files in `data/raw/`. Ensure filenames, column
names, and formats match the dashboard's expected schema.

## Installation

Activate the existing Conda environment:

``` powershell
conda activate retailpulse
```

Install dependencies from the project root:

``` powershell
python -m pip install -r requirements.txt
```

If the requirements file is incomplete, install the dashboard's core
packages:

``` powershell
python -m pip install pandas numpy streamlit plotly
```

## Run the dashboard

From the project root:

``` powershell
conda activate retailpulse
cd frontend
python -m streamlit run app.py
```

Open the local address printed by Streamlit, normally
`http://localhost:8501`.

The current standalone dashboard reads processed CSV files directly.
Start the FastAPI backend only if the specific feature/version you run
explicitly requires it.

## Basic usage

1.  Start Streamlit.
2.  Set a date range that includes dates in the dataset.
3.  Select the required country or countries.
4.  Leave Customer ID empty to view all customers, or enter an exact ID
    copied from `customers.csv`.
5.  Select the required sales channel and currency view.
6.  Navigate through the 12 workspace pages.

If no records appear, clear Customer ID, widen the date range, and
select all countries and channels. A no-results message means the
current filter combination did not match records; it does not
necessarily mean the dataset is empty.

## Data and analytics limitations

-   The supplied demonstration dataset is synthetic. Customer profiles,
    contact details, transactions, orders, payments, shipments, returns,
    campaigns, and inventory values are simulated.
-   Simulated transaction updates are demonstrations, not live retailer
    transactions.
-   Inventory is a product-level snapshot, not a live inventory
    integration.
-   Inactivity-risk indicators are proxies based on available purchase
    history unless a separately trained and validated churn model is
    integrated.
-   Estimated profit uses the cost fields in the supplied dataset. It is
    not audited accounting profit.
-   Multiple currencies are present. Do not aggregate amounts across
    currencies without documented conversion.
-   Demand forecasting, churn prediction, product recommendation, and
    PostgreSQL integration should only be described as implemented if
    their models or database have actually been integrated and tested.

## Troubleshooting

### Streamlit command is not recognized

Use:

``` powershell
python -m streamlit run app.py
```

### Missing Python imports

Activate the correct environment and install dependencies:

``` powershell
conda activate retailpulse
python -m pip install -r requirements.txt
```

In VS Code, select the Conda interpreter at:

``` text
C:\Users\<your-user>\anaconda3\envs\retailpulse\python.exe
```

### Dashboard shows zero records

-   Clear the Customer ID filter.
-   Widen the date range to cover the dataset.
-   Select all countries and sales channels.
-   Confirm required CSV files exist in `data/processed/`.
-   Confirm the Customer ID exists in `customers.csv`.

### CSV loading errors

Check filenames, column names, date formats, and data types against the
expected dataset schema. Do not mix unrelated old and new CSV files in
`data/processed/`.

## Project status

This README documents the CSV-driven analytics dashboard and its
demonstration-data setup. Additional machine-learning models, live
integrations, or database features must be verified against the actual
code and deployment configuration before being presented as completed.

## License and data handling

Add the applicable project/client license and data-handling terms before
external distribution. Do not publish real customer personal
information, credentials, API keys, or confidential client data.
