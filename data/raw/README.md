# RetailPulse Complete Dataset

**Synthetic demonstration data only**: not real customers, not sourced from a retailer, and not a live feed. Names/contact details/purchases/payments/shipments/inventory are generated; email uses example.com.

## Scale
10,000 customers; 1,000 products; 50,000 orders; more than 50,000 purchase lines; payment, shipment, return, inventory, campaign and customer analytics tables.

## Recommended dashboard sources
`data/transactions_flat.csv` (main sales fact); `data/customer_analytics_rfm.csv` (customer analytics); `data/product_performance.csv`; `data/orders.csv`; `data/order_items.csv`; `data/payments.csv`; `data/shipments.csv`; `data/returns.csv`; `data/inventory_snapshot.csv`; `data/campaigns.csv`; `data/live_transactions_seed.csv` (simulated-live demo).

## Joins
customers.customer_id -> orders.customer_id; orders.order_id -> order_items.order_id/payments.order_id/shipments.order_id; order_items.product_id -> products.product_id; order_items.order_item_id -> returns.order_item_id.

## Metric cautions
Estimated profit uses generated unit costs, not audited accounting. Inventory is a static generated snapshot. Inactivity risk is a recency proxy, not verified churn. The live seed is not real-time. Data has INR and USD; filter/convert currencies before summing. All records are synthetic.
