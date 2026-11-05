\# RetailPulse – AI-Powered Customer Analytics \& Demand Forecasting Platform



RetailPulse is an AI-powered retail analytics and predictive intelligence platform designed to help businesses understand customer behaviour, monitor sales performance, forecast demand, identify customer inactivity risks, and optimize inventory decisions.



The platform combines an interactive Streamlit dashboard, machine learning models, PostgreSQL data storage, Redis caching, and MLflow experiment tracking.



> \*\*Dataset Notice:\*\* The demonstration dataset used in this project contains synthetic records generated for development, testing, and demonstration. It does not represent real customer or retailer transaction data. All business insights and model outputs should be interpreted as results from simulated data, not actual business performance.



\---



\## 1. Project Objectives



RetailPulse aims to:



\- Analyse retail sales and revenue performance.

\- Understand customer purchasing behaviour.

\- Identify meaningful customer segments using RFM and clustering.

\- Forecast future retail demand.

\- Identify customers with potential inactivity risk.

\- Generate inventory reorder recommendations.

\- Provide an interactive dashboard for business intelligence.

\- Track machine learning experiments and model performance.



\---



\## 2. Dashboard Features



The Streamlit dashboard contains \*\*14 workspace pages\*\*.



| No. | Workspace Page | Description |

|---|---|---|

| 1 | Executive Summary | Overall retail business performance and key indicators. |

| 2 | Sales Trend Analysis | Sales trends and revenue patterns over time. |

| 3 | Product Performance | Product-level sales and performance analysis. |

| 4 | Customer Analytics | Customer metrics and purchasing insights. |

| 5 | Region \& Country Sales | Geographical sales distribution and country comparisons. |

| 6 | Monthly \& Seasonal Sales | Monthly trends and seasonal patterns. |

| 7 | Customer Behaviour | Customer purchasing patterns and behaviour analysis. |

| 8 | Revenue \& Profit Analysis | Revenue and estimated profit analysis. |

| 9 | Inventory Risk | Inventory status, stock coverage, and reorder recommendations. |

| 10 | Orders \& Transactions | Order and transaction-level analysis. |

| 11 | Advanced Analytics | Additional analytical insights and interactive visualizations. |

| 12 | Interactive Filters | Interactive filtering of dashboard data. |

| 13 | AI \& Predictive Analytics | Machine learning-based analytical modules. |

| 14 | Model Performance \& Tracking | Model evaluation and experiment tracking information. |



\### AI \& Predictive Analytics Modules



The AI \& Predictive Analytics workspace contains four modules:



1\. \*\*Customer Segmentation\*\* – RFM analysis and K-Means clustering to identify customer groups.

2\. \*\*Demand Forecasting\*\* – Prophet and LSTM-based demand forecasting experiments.

3\. \*\*Churn Analytics\*\* – Customer inactivity risk analysis and predictive model evaluation.

4\. \*\*Inventory Optimization\*\* – Stock coverage analysis and reorder recommendations.



\### Global Dashboard Filters



The dashboard provides filters for:



\- Order date range.

\- Country.

\- Customer ID.

\- Sales channel.

\- Currency view, where supported.



Customer-linked records are filtered through the selected customer and related orders. Inventory is product-level data and is not associated with an individual customer.



Amounts in different currencies should not be combined without a documented exchange-rate conversion.



\---



\## 3. Technology Stack



\### Programming and Analytics



\- Python 3.11

\- Pandas

\- NumPy

\- Scikit-learn



\### Dashboard and Visualization



\- Streamlit

\- Plotly

\- Matplotlib



\### Machine Learning and Forecasting



\- Scikit-learn

\- Prophet

\- PyTorch

\- LSTM

\- K-Means Clustering

\- Random Forest



\### Database and Caching



\- PostgreSQL 16

\- Redis 7



\### Experiment Tracking



\- MLflow



\### Development and Deployment Tools



\- Anaconda

\- Git

\- GitHub

\- Docker Desktop



\---



\## 4. Project Structure



```text

RetailPulse\_AI/

│

├── backend/

│   ├── \_\_init\_\_.py

│   ├── analytics.py

│   ├── config.py

│   ├── data\_loader.py

│   └── main.py

│

├── data/

│   ├── raw/

│   └── processed/

│

├── frontend/

│   ├── app.py

│   ├── app\_backup.py

│   ├── api\_client.py

│   ├── db\_loader.py

│   └── components/

│

├── models/

│   ├── churn\_prediction.py

│   ├── customer\_segmentation.py

│   ├── demand\_forecasting.py

│   ├── daily\_lstm\_demand\_forecasting.py

│   ├── lstm\_demand\_forecasting.py

│   ├── inventory\_optimization.py

│   ├── final\_customer\_segmentation.py

│   └── outputs/

│

├── scripts/

│   ├── inspect\_mlflow\_dbs.py

│   ├── integrate\_database.py

│   ├── load\_to\_postgres.py

│   └── redis\_test.py

│

├── performance\_test.py

├── scalability\_test.py

├── requirements.txt

├── README.md

└── .gitignore

```



The `models/outputs/` directory contains generated model results, evaluation metrics, forecasts, and other analytical artifacts.



\---



\## 5. Dataset Information



The project uses processed retail demonstration datasets.



Common datasets include:



\- `customers.csv`

\- `products.csv`

\- `orders.csv`

\- `order\_items.csv`

\- `payments.csv`

\- `shipments.csv`

\- `returns.csv`

\- `inventory\_snapshot.csv`

\- `campaigns.csv`

\- `transactions\_flat.csv`

\- `customer\_analytics\_rfm.csv`

\- `product\_performance.csv`

\- `country\_sales\_summary.csv`

\- `daily\_sales\_summary.csv`

\- `monthly\_sales\_summary.csv`

\- `payment\_summary.csv`

\- `live\_transactions\_seed.csv`



\### Data Handling



\- Raw source files are maintained in `data/raw/`.

\- Processed datasets are maintained in `data/processed/`.

\- Processed data is loaded into PostgreSQL for dashboard access.

\- Redis is used for caching selected data.

\- Data preparation and loading scripts are maintained in `scripts/`.



\---



\## 6. Database Architecture



RetailPulse uses PostgreSQL and Redis as part of its data infrastructure.



\### PostgreSQL



PostgreSQL is used for structured retail data storage.



The project includes tables for:



\- Customers

\- Products

\- Orders

\- Order items

\- Payments

\- Shipments

\- Returns

\- Inventory

\- Campaigns

\- Sales summaries

\- Customer analytics



\### Redis



Redis provides caching for selected dashboard data to reduce repeated database queries.



\### Database Integration Files



| File | Purpose |

|---|---|

| `frontend/db\_loader.py` | Loads data from the database and supports caching. |

| `scripts/load\_to\_postgres.py` | Loads processed CSV datasets into PostgreSQL. |

| `scripts/redis\_test.py` | Tests Redis connectivity and caching. |

| `backend/config.py` | Contains backend configuration. |

| `scripts/integrate\_database.py` | Supports database integration into the dashboard. |



\### Database Requirements



\- PostgreSQL 16

\- Redis 7

\- Docker Desktop, if using the project's Docker-based local services.



Configure database connection details locally according to the project's configuration. Never publish passwords, API keys, or other credentials in the repository.



\---



\## 7. Machine Learning Modules



\### 7.1 Customer Segmentation



Customer segmentation uses RFM-related customer behaviour features and K-Means clustering.



The final segmentation experiment uses six clusters with business-oriented interpretations:



1\. Hibernating Customers

2\. At-Risk Customers

3\. Recent Customers

4\. High-Value Loyalists

5\. Champions

6\. Potential Loyalists



The final segmentation experiment produced a silhouette score of approximately \*\*0.2636\*\*.



The six-cluster solution was selected for business interpretability and segment granularity, rather than because it achieved the highest silhouette score.



Segment labels are descriptive interpretations of customer profiles, not independently validated future customer outcomes.



\### 7.2 Demand Forecasting



Demand forecasting experiments include:



\- Prophet-based forecasting.

\- LSTM-based forecasting.

\- Weekly rolling validation.

\- Daily forecasting.

\- Model comparison and evaluation.



Forecasting results are evaluated using metrics such as:



\- MAE

\- RMSE

\- MAPE

\- WAPE



\*\*Important evaluation note:\*\* Forecasting performance varies by currency, frequency, and validation method.



In the weekly rolling validation experiments:



\- INR Random Forest achieved approximately 7.86% MAPE.

\- INR Prophet achieved approximately 8.53% MAPE.

\- USD LSTM achieved approximately 12.19% MAPE.



The Zidio target of MAPE ≤12% has not been achieved across all evaluated currencies and models.



Daily forecasting experiments also produced higher errors. Weekly validation results must not be presented as daily forecasting accuracy.



\### 7.3 Churn Analytics



The churn module analyses customer inactivity risk using available purchasing history.



The leakage-free baseline Random Forest experiment achieved approximately:



\- ROC-AUC: 0.687



The target ROC-AUC of 0.88 has not been achieved.



The current module should be interpreted as an experimental customer inactivity prediction capability, not a fully validated future churn prediction system.



\### 7.4 Inventory Optimization



The inventory module analyses stock availability and generates reorder recommendations.



The current demonstration logic uses:



\- Stock on hand.

\- Average daily sales.

\- Days of inventory coverage.

\- Reorder levels.

\- Target stock quantities.



Inventory categories include:



\- Critical

\- Low Stock

\- Moderate

\- Healthy



The current reorder calculation uses a 30-day target stock horizon.



Lead times, minimum order quantities, and advanced safety-stock calculations are not fully modelled.



\---



\## 8. Model Performance and Experiment Tracking



MLflow is used for recording machine learning experiments and evaluation metrics.



The project includes experiment tracking for demand forecasting models.



Tracked metrics include:



\- MAE

\- RMSE

\- MAPE

\- WAPE



MLflow tracking is implemented locally. Production model registry, automated monitoring, and automated retraining are not claimed as completed deployment capabilities.



\---



\## 9. Installation and Environment Setup



\### Prerequisites



Install:



1\. Python 3.11 / Anaconda

2\. Git

3\. Docker Desktop

4\. A supported web browser



\### Step 1: Clone the Repository



Open Windows PowerShell:



```powershell

git clone https://github.com/Gowthami-max/RetailPulse-AI.git



cd RetailPulse-AI

```



\### Step 2: Activate the Conda Environment



If the environment already exists:



```powershell

conda activate retailpulse

```



For a fresh environment:



```powershell

conda create -n retailpulse python=3.11 -y



conda activate retailpulse

```



\### Step 3: Install Dependencies



From the project root:



```powershell

python -m pip install --upgrade pip



python -m pip install -r requirements.txt

```



If a required dashboard package is missing:



```powershell

python -m pip install pandas numpy streamlit plotly scikit-learn statsmodels

```



Install additional model dependencies only if required by the specific module being executed.



\---



\## 10. Database Setup



Start Docker Desktop before starting the database services.



The local development setup uses:



\- PostgreSQL container: `retailpulse-postgres`

\- Redis container: `retailpulse-redis`



Check running containers:



```powershell

docker ps

```



The local PostgreSQL database is named `retailpulse\_db`.



Ensure the database connection settings in the project configuration match the local PostgreSQL and Redis services.



Load the processed datasets into PostgreSQL using:



```powershell

python scripts/load\_to\_postgres.py

```



Verify Redis connectivity using:



```powershell

python scripts/redis\_test.py

```



Do not run the database loading script unnecessarily because it replaces existing tables with the same names.



\---



\## 11. Run the RetailPulse Dashboard



From the project root:



```powershell

conda activate retailpulse



cd C:\\Users\\ASUS\\Desktop\\RetailPulse\_AI



python -m streamlit run frontend/app.py

```



Streamlit will display a local URL, normally:



```text

http://localhost:8501

```



If port 8501 is already occupied, Streamlit may select another available port.



Open the displayed URL in your browser.



\---



\## 12. Running Individual Machine Learning Modules



Machine learning scripts are available inside the `models/` directory.



Examples:



```powershell

python models/final\_customer\_segmentation.py

```



```powershell

python models/demand\_forecasting.py

```



```powershell

python models/daily\_lstm\_demand\_forecasting.py

```



```powershell

python models/inventory\_optimization.py

```



Run scripts from the project root so that their relative file paths resolve correctly.



Check each script's configuration and dependencies before execution.



\---



\## 13. Performance and Scalability Testing



The project includes performance and scalability test scripts.



\### Performance Test



```powershell

python performance\_test.py

```



\### Scalability Test



```powershell

python scalability\_test.py

```



The scalability experiment processes 10 million synthetic records in batches and records the processing results.



The benchmark is a local synthetic-data experiment. It does not demonstrate production or distributed-cluster throughput.



\---



\## 14. Data and Model Limitations



The following limitations should be considered when interpreting project results:



1\. The demonstration dataset is synthetic and does not contain actual retailer transaction records.

2\. Simulated transaction updates are not live retailer transactions.

3\. Inventory data represents a product-level snapshot, not a live inventory integration.

4\. Customer inactivity indicators are based on available purchase history.

5\. Churn model performance is below the Zidio target of ROC-AUC ≥0.88.

6\. Forecasting performance differs across currencies and validation frequencies.

7\. The MAPE ≤12% forecasting target is not achieved across all evaluated models and currencies.

8\. The inventory recommendation logic does not fully account for supplier lead times, minimum order quantities, or advanced safety stock.

9\. Estimated profit is based on supplied demonstration cost fields and is not audited accounting profit.

10\. Currency values should not be combined without documented exchange-rate conversion.

11\. MLflow tracking is implemented locally; production model registry and automated retraining are not fully implemented.

12\. The scalability benchmark is based on synthetic data and a local environment.



These limitations are documented to distinguish experimental results from validated production capabilities.



\---



\## 15. Troubleshooting



\### Streamlit Command Not Recognized



Use:



```powershell

python -m streamlit run frontend/app.py

```



\### Missing Python Packages



Activate the Conda environment:



```powershell

conda activate retailpulse



python -m pip install -r requirements.txt

```



\### Database Connection Error



Check:



\- Docker Desktop is running.

\- PostgreSQL container is running.

\- Redis container is running.

\- Database credentials and connection settings are correct.

\- Required database tables have been loaded.



\### Dashboard Shows No Records



Check:



\- Database connectivity.

\- Required datasets and tables.

\- Selected date range.

\- Country and sales-channel filters.

\- Customer ID selection.



\### Forecast Files Not Found



Check that the required model output files exist under:



```text

models/outputs/forecasting/

```



Run the corresponding forecasting script if the required outputs have not been generated.



\---



\## 16. Future Enhancements



Potential improvements include:



\- Improving churn prediction using additional validated behavioural features.

\- Improving forecasting accuracy across currencies.

\- Developing a validated Prophet-LSTM ensemble.

\- Incorporating supplier lead times and safety-stock calculations.

\- Implementing automated data quality monitoring.

\- Adding model drift detection and automated retraining.

\- Integrating Prometheus and Grafana monitoring.

\- Extending deployment with Kubernetes.

\- Connecting real retailer data sources through secure integrations.



\---



\## 17. Project Repository



\*\*GitHub Repository:\*\*



https://github.com/Gowthami-max/RetailPulse-AI



\---



\## 18. License and Data Handling



This project is developed as a retail analytics and machine learning demonstration platform.



Add the applicable client/project license and data-handling terms before external distribution.



Do not publish:



\- Real customer personal information.

\- Database passwords.

\- API keys.

\- Authentication tokens.

\- Confidential client information.



The repository's `.gitignore` excludes selected local databases, credentials, and generated benchmark artifacts. Review files before making future commits.



\---



\*\*RetailPulse – Turning Retail Data into Actionable Intelligence.\*\*
