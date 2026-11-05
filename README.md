# RetailPulse — AI-Powered Customer Analytics & Demand Forecasting Platform

> **An end-to-end retail analytics demonstration for sales intelligence, customer segmentation, demand forecasting, inactivity-risk analysis, and inventory decision support.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://retailpulse-ai-r7xsewulhdyqfxzxkrzfxe.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Database](https://img.shields.io/badge/Database-Neon%20PostgreSQL-008CC1)](https://neon.tech/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)

**Project:** RetailPulse  
**Prepared by:** Gowthami  
**Prepared for:** Zidio Development — Data Science & Analytics  
**Repository:** https://github.com/Gowthami-max/RetailPulse-AI  
**Live application:** https://retailpulse-ai-r7xsewulhdyqfxzxkrzfxe.streamlit.app

---

## 1. Project Overview

RetailPulse brings retail transaction, customer, product, and inventory information into one interactive analytics application. It combines data preparation, analytical summaries, machine-learning experiment outputs, a relational database, and a public Streamlit dashboard.

The platform is designed to help users explore questions such as:

- How are sales and order volumes changing over time?
- Which products, categories, and countries contribute most to sales?
- Which customer groups show different purchasing patterns?
- What do the available forecasting models estimate for future demand?
- Which customers are flagged as potentially inactive?
- Which products are flagged for inventory review or reorder?

**Data notice:** The repository's demonstration dataset is synthetic. Dashboard values are for project demonstration and technical evaluation; they are not actual retailer transactions or independently audited business outcomes.

## 2. Key Capabilities

| Capability | What the application provides |
|---|---|
| Data preparation | Data inspection, cleaning, transformation, and analytical table generation |
| Sales analytics | Executive KPIs, daily/weekly/monthly trends, revenue and order analysis |
| Product analytics | Product and category performance summaries |
| Customer analytics | Customer behaviour, RFM-style features, and segment summaries |
| Customer segmentation | K-Means clustering with six business-interpreted groups |
| Demand forecasting | Weekly and daily forecast comparisons using Random Forest, Prophet, and LSTM experiment outputs |
| Inactivity-risk analysis | Baseline classification and customer inactivity-risk summaries |
| Inventory optimization | Stock-risk categories and estimated reorder recommendations |
| Interactive dashboard | Streamlit pages, Plotly charts, filters, tables, and selected CSV exports |
| Data storage | Neon PostgreSQL for the deployed relational data layer; Redis caching is used in the application setup |

## 3. Live Demo

**Open the deployed application:**  
https://retailpulse-ai-r7xsewulhdyqfxzxkrzfxe.streamlit.app

The public dashboard contains 14 workspace pages:

1. Executive Summary
2. Sales Trend Analysis
3. Product Performance
4. Customer Analytics
5. Region & Country Sales
6. Monthly & Seasonal Sales
7. Customer Behaviour
8. Revenue & Profit Analysis
9. Inventory Risk
10. Orders & Transactions
11. Advanced Analytics
12. Interactive Filters
13. AI & Predictive Analytics
14. Model Performance & Tracking

The **AI & Predictive Analytics** area contains four modules: Customer Segmentation, Demand Forecasting, Churn Analytics, and Inventory Optimization.

## 4. Technology Stack

| Layer | Technologies |
|---|---|
| Language | Python 3.11 |
| Data processing | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Forecasting experiments | Prophet, PyTorch / LSTM |
| Dashboard | Streamlit |
| Visualizations | Plotly |
| Relational database | PostgreSQL / Neon |
| Caching | Redis |
| Database connectivity | SQLAlchemy, psycopg2 |
| Deployment | Streamlit Community Cloud |
| Version control | Git, GitHub |

## 5. High-Level Architecture

```text
Synthetic retail source datasets (CSV)
                 |
                 v
       Data inspection & cleaning
                 |
                 v
     Feature engineering / summaries
                 |
        +--------+---------+
        |                  |
        v                  v
  ML experiments      Analytical outputs
  - Segmentation      - Sales summaries
  - Forecasting       - Customer summaries
  - Inactivity risk   - Inventory outputs
        |                  |
        +--------+---------+
                 |
                 v
       PostgreSQL / Neon data layer
                 |
          Data loader + cache
                 |
                 v
        Streamlit + Plotly dashboard
                 |
                 v
     Interactive analysis / CSV exports
```

The diagram represents the project's logical data flow. Model outputs and analytical summaries are generated by project workflows and surfaced by the dashboard. The deployed application should not be interpreted as a continuously retraining or fully real-time production ML system.

## 6. Data & Database

The deployed dashboard is connected to a Neon PostgreSQL database. The application loader makes 18 tables available to the dashboard, including operational entities and derived summaries.

Examples of the data entities include:

- `customers`
- `products`
- `orders`
- `order_items`
- `payments`
- `shipments`
- `returns`
- `inventory_snapshot`
- `campaigns`
- `transactions_flat`
- `customer_analytics_rfm`
- `product_performance`
- `country_sales_summary`
- `daily_sales_summary`
- `monthly_sales_summary`
- `payment_summary`
- `live_transactions_seed`

The core demonstration tables include approximately 50,000 orders, 10,000 customers, and 1,000 products.

**Important:** INR and USD are treated as separate currency groups. Do not add the two currencies together as if they were a single currency.

## 7. Model Results & Evaluation

The following values are the project-reported experiment results. They should be interpreted within the evaluation setup stated below.

### Customer segmentation

| Measure | Result |
|---|---:|
| Algorithm | K-Means |
| Selected clusters | 6 |
| Silhouette score | 0.2636 |
| Davies–Bouldin index | 1.0596 |
| Calinski–Harabasz index | 5,513.97 |

Six groups were selected for business interpretability. The silhouette score indicates moderate/limited cluster separation; segment labels are heuristic interpretations, not independently validated customer personas.

### Demand forecasting

| Frequency / currency | Model | MAPE |
|---|---|---:|
| Weekly INR | Random Forest | 7.86% |
| Weekly INR | Prophet | 8.53% |
| Weekly USD | LSTM | 12.19% |
| Weekly USD | Random Forest | 12.87% |
| Weekly USD | Prophet | 13.97% |
| Daily INR | Prophet | 24.04% |
| Daily INR | LSTM | 24.17% |
| Daily USD | Prophet | 28.51% |
| Daily USD | LSTM | 29.75% |

The 12% MAPE target is met by the reported weekly INR Random Forest and Prophet results, but not across all currency/frequency/model combinations. Forecasts are exploratory estimates and not observed future sales.

### Customer inactivity / churn baseline

| Measure | Result |
|---|---:|
| Inactivity definition | 180 or more days since last purchase |
| Leakage-controlled baseline ROC-AUC | Approximately 0.6865 |
| Zidio target | ROC-AUC ≥ 0.88 |

The reported baseline is below the target. It is presented as a baseline rather than a production-grade churn model.

### Inventory optimization

The deployed dashboard displays approximately 1,000 products tracked, 45 critical products, and 117 products flagged for reorder. Estimated reorder costs and quantities are generated recommendations, not supplier quotes or approved purchase orders. The current logic does not fully model verified supplier lead times, minimum order quantities, or service-level safety stock.

### Processing and scale benchmarks

- Saved data-processing benchmark: 8 files processed in approximately 1.587 seconds against a 300-second target in the tested run.
- Local synthetic scalability benchmark: approximately 10 million rows processed in about 33.3 seconds on the test machine.

These are specific project benchmarks, not cloud production SLAs or distributed throughput guarantees. Processing time is not a machine-learning accuracy metric.

## 8. Run Locally

### Prerequisites

- Windows, macOS, or Linux
- Python 3.11
- Git
- Access to the required project data files
- PostgreSQL and Redis if you intend to reproduce the local database/cache configuration

### Clone the repository

```bash
git clone https://github.com/Gowthami-max/RetailPulse-AI.git
cd RetailPulse-AI
```

### Create and activate the Conda environment

```bash
conda create -n retailpulse python=3.11 -y
conda activate retailpulse
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment variables

Create a local `.env` file in the project root. Do not commit this file.

Example structure (replace placeholders with your own database settings):

```env
DATABASE_URL=postgresql+psycopg2://<user>:<password>@<host>:5432/<database>
NEON_DATABASE_URL=<your-Neon-connection-string>
REDIS_HOST=localhost
REDIS_PORT=6379
```

Use the connection string format and SSL settings provided by your database provider. If using Streamlit Community Cloud, configure the required values through the app's Secrets settings rather than committing credentials.

**Security:** Never paste database passwords, API keys, or complete connection strings into source code, README files, screenshots, or public Git commits.

### Start the dashboard

From the repository root:

```bash
python -m streamlit run frontend/app.py
```

Streamlit will print a local address, normally `http://localhost:8501`.

> The dashboard expects the project data/output files and database tables referenced by the application to be present. A fresh clone alone may not contain private credentials or every locally generated artifact. Configure the data layer before expecting all pages to populate.

## 9. Deployment

The public demonstration is hosted on Streamlit Community Cloud and uses the GitHub `main` branch. The deployed database is Neon PostgreSQL.

Deployment outline:

1. Push the application source to the GitHub repository.
2. Create or select the Streamlit Community Cloud app.
3. Set the repository, branch (`main`), and main file (`frontend/app.py`).
4. Select a compatible Python version.
5. Add database configuration through Streamlit Secrets.
6. Deploy and inspect application logs.
7. Verify the dashboard using the public HTTPS URL.

Do not expose `.env` files or secrets in GitHub. The local development PostgreSQL/Redis containers are not the same thing as the hosted Neon database.

## 10. Repository Structure

The project is organized around the following major areas:

```text
RetailPulse-AI/
├── frontend/
│   ├── app.py
│   └── db_loader.py
├── backend/
├── data/
├── models/
│   └── outputs/
├── reports/
├── scripts/
├── requirements.txt
├── .gitignore
└── README.md
```

The exact contents of `data/`, `models/outputs/`, `reports/`, and `scripts/` depend on the generated artifacts and project version. Refer to the repository for the current file tree.

## 11. Known Limitations & Responsible Use

- Demonstration data is synthetic; results do not represent an actual retailer.
- Forecast accuracy varies by currency, model and time frequency.
- The churn baseline ROC-AUC is below the stated target.
- Inventory recommendations are estimates and require operational constraints and live stock validation.
- Profit-related fields may use generated unit-cost assumptions and should not be treated as audited financial margin.
- The project does not claim a production remote MLflow tracking server, automated retraining loop, or continuous drift-monitoring service.
- Local benchmark results are not guarantees of cloud performance.
- The public demo is not a hardened enterprise deployment with comprehensive role-based access control.

Use the dashboard as an analytical demonstration and decision-support prototype. Validate all model outputs and inventory recommendations before using them for real commercial decisions.

## 12. Future Improvements

1. Improve churn prediction with temporal validation, probability calibration, and top-decile lift evaluation.
2. Extend rolling-origin forecast evaluation and tune models separately for each currency and forecast horizon.
3. Incorporate verified supplier lead times, minimum order quantities, and safety-stock policies into inventory optimization.
4. Add automated tests for data schemas, ETL workflows, database loading, and critical dashboard pages.
5. Configure managed experiment tracking, model versioning, and a controlled model promotion process.
6. Add data/model drift monitoring and an approved retraining workflow.
7. Strengthen production security, access control, backup, and operational monitoring.
8. Validate business outcomes through a controlled pilot using real, authorized retail data.

## 13. Project Links

- **Live dashboard:** https://retailpulse-ai-r7xsewulhdyqfxzxkrzfxe.streamlit.app
- **GitHub repository:** https://github.com/Gowthami-max/RetailPulse-AI
- **Streamlit:** https://streamlit.io/
- **Neon:** https://neon.tech/

---

## Acknowledgement

Prepared by **Gowthami** for the **Zidio Development — Data Science & Analytics** project submission.

*RetailPulse — turning retail data into clearer operational insight.*
