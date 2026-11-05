from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

# ============================================================
# RetailPulse — Dataset-aligned dashboard
# Works with the RetailPulse Complete Dataset CSV schema.
# Put CSVs in ../data/processed OR ../data relative to this file.
# ============================================================

st.set_page_config(
    page_title="RetailPulse | Customer & Commerce Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background: radial-gradient(circle at 15% 0%, #14233a 0%, #0b1220 38%, #080d17 100%); color:#edf3ff;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#111c2d,#0b1220); border-right:1px solid #24354d;}
[data-testid="stSidebar"] * {color:#e8eef9;}
h1,h2,h3 {font-family:'Manrope',sans-serif; letter-spacing:-.4px;}
h1 {font-size:2.15rem!important; font-weight:800!important;}
.block-container {padding-top:1.35rem; padding-bottom:2rem; max-width:1600px;}
[data-testid="stMetric"] {background:linear-gradient(145deg,#17263b,#101b2b); border:1px solid #2a3d57; padding:16px 18px; border-radius:15px; box-shadow:0 8px 24px rgba(0,0,0,.15);}
[data-testid="stMetricLabel"] {color:#9fb1c9!important;}
[data-testid="stMetricValue"] {color:#f4f7fc!important; font-weight:800;}
[data-testid="stMetricDelta"] {color:#52d6a2!important;}
div[data-testid="stDataFrame"] {border:1px solid #293b54; border-radius:12px; overflow:hidden;}
.stTabs [data-baseweb="tab-list"] {gap:8px;}
.stTabs [data-baseweb="tab"] {background:#152238; border-radius:9px 9px 0 0; padding:9px 16px;}
div.stButton>button {border-radius:9px; border:1px solid #355071; background:#1b3552; color:white;}
div.stButton>button:hover {border-color:#58b9ff; color:white;}
.rp-card {background:linear-gradient(145deg,#17263b,#101a29);border:1px solid #2a3d57;border-radius:16px;padding:18px 20px;margin-bottom:10px;}
.rp-muted {color:#9fb1c9;font-size:.88rem;}
.rp-pill {display:inline-block;background:#173c3a;color:#76e4c0;border:1px solid #27685e;padding:4px 10px;border-radius:30px;font-size:.78rem;}
hr {border-color:#26364c;}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#dce7f5"),
    margin=dict(l=12,r=12,t=45,b=12),
    legend=dict(orientation="h",y=1.12),
)

REQUIRED_FILES = [
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

D = get_all()
orders = D["orders"].copy()
customers = D["customers"].copy()
items = D["order_items"].copy()
products = D["products"].copy()
flat = D["transactions_flat"].copy()
payments = D["payments"].copy()
shipments = D["shipments"].copy()
returns = D["returns"].copy()
inventory = D["inventory_snapshot"].copy()
campaigns = D["campaigns"].copy()
cust_analytics = D["customer_analytics_rfm"].copy()
prod_perf = D["product_performance"].copy()

# Data normalization
if not orders.empty:
    orders["order_date"] = pd.to_datetime(orders.get("order_date"), errors="coerce")
    orders["order_datetime"] = pd.to_datetime(orders.get("order_datetime"), errors="coerce")
if not flat.empty:
    flat["order_date"] = pd.to_datetime(flat.get("order_date"), errors="coerce")
    flat["order_datetime"] = pd.to_datetime(flat.get("order_datetime"), errors="coerce")

# Sidebar / global filters
st.sidebar.markdown("## 🛍️ RetailPulse")
st.sidebar.caption("Customer & Commerce Intelligence")
st.sidebar.markdown("---")
pages = [
    "Executive Summary", "Sales Trend Analysis", "Product Performance",
    "Customer Analytics", "Region & Country Sales", "Monthly & Seasonal Sales",
    "Customer Behaviour", "Revenue & Profit Analysis", "Inventory Risk",
    "Orders & Transactions", "Advanced Analytics", "Interactive Filters",
    "AI & Predictive Analytics","Model Performance & Tracking",
]
page = st.sidebar.radio("WORKSPACE", pages, index=0)
st.sidebar.markdown("---")

if orders.empty:
    st.error("orders.csv is empty or could not be loaded.")
    st.stop()

min_date = orders["order_date"].min().date()
max_date = orders["order_date"].max().date()
default_start = max(min_date, max_date - timedelta(days=365))
date_range = st.sidebar.date_input("Order date range", value=(default_start, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range)==2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

countries = sorted(orders["shipping_country"].dropna().astype(str).unique()) if "shipping_country" in orders else []
selected_countries = st.sidebar.multiselect("Country", countries, default=[])
customer_ids = sorted(customers["customer_id"].dropna().astype(str).unique()) if "customer_id" in customers else []
customer_search = st.sidebar.text_input("Customer ID (exact match)", placeholder="e.g. CUST000123").strip().upper()
channels = sorted(orders["sales_channel"].dropna().astype(str).unique()) if "sales_channel" in orders else []
selected_channels = st.sidebar.multiselect("Sales channel", channels, default=[])
st.sidebar.markdown("---")
st.sidebar.caption("Data source: PostgreSQL + Redis")
if st.sidebar.button("↻ Refresh dataset"):
    st.cache_data.clear()
    st.rerun()

# Exact-match global filtering: never silently fall back to all customers.
mask = orders["order_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1)-pd.Timedelta(seconds=1))
if selected_countries:
    mask &= orders["shipping_country"].isin(selected_countries)
if selected_channels:
    mask &= orders["sales_channel"].isin(selected_channels)
if customer_search:
    mask &= orders["customer_id"].astype(str).str.upper().eq(customer_search)
f_orders = orders.loc[mask].copy()
selected_order_ids = set(f_orders["order_id"].astype(str))
f_items = items[items["order_id"].astype(str).isin(selected_order_ids)].copy()
f_flat = flat[flat["order_id"].astype(str).isin(selected_order_ids)].copy() if not flat.empty and "order_id" in flat else pd.DataFrame()
f_payments = payments[payments["order_id"].astype(str).isin(selected_order_ids)].copy()
f_shipments = shipments[shipments["order_id"].astype(str).isin(selected_order_ids)].copy()
f_returns = returns[returns["order_id"].astype(str).isin(selected_order_ids)].copy()
f_customers = customers[customers["customer_id"].astype(str).isin(f_orders["customer_id"].astype(str).unique())].copy()
f_inventory = inventory.copy()
f_products = products.copy()

# Currency safety: metrics are split by currency to avoid mixing INR and USD.
currency_values = sorted(f_orders["currency"].dropna().astype(str).unique()) if "currency" in f_orders else []
if len(currency_values) > 1:
    currency_choice = st.sidebar.selectbox("Currency view", ["All currencies (separate totals)"] + currency_values)
else:
    currency_choice = currency_values[0] if currency_values else "All currencies"
if currency_choice != "All currencies (separate totals)" and currency_choice != "All currencies" and "currency" in f_orders:
    f_orders = f_orders[f_orders["currency"].eq(currency_choice)]
    selected_order_ids = set(f_orders["order_id"].astype(str))
    f_items = items[items["order_id"].astype(str).isin(selected_order_ids)].copy()
    f_flat = flat[flat["order_id"].astype(str).isin(selected_order_ids)].copy() if not flat.empty else pd.DataFrame()
    f_payments = payments[payments["order_id"].astype(str).isin(selected_order_ids)].copy()
    f_shipments = shipments[shipments["order_id"].astype(str).isin(selected_order_ids)].copy()
    f_returns = returns[returns["order_id"].astype(str).isin(selected_order_ids)].copy()
    f_customers = customers[customers["customer_id"].astype(str).isin(f_orders["customer_id"].astype(str).unique())].copy()

# Helper functions
def money(v, currency=None):
    if pd.isna(v): v=0
    return (f"{currency} " if currency else "") + f"{float(v):,.2f}"

def metric_row():
    n_orders = f_orders["order_id"].nunique()
    n_customers = f_orders["customer_id"].nunique()
    revenue = f_orders.loc[~f_orders["order_status"].isin(["Cancelled","Returned"]), "order_total"].sum()
    aov = revenue / max(n_orders,1)
    completed = (f_orders["order_status"]=="Delivered").sum()
    returns_n = f_returns["return_id"].nunique() if "return_id" in f_returns else len(f_returns)
    ccy = currency_choice if currency_choice not in ["All currencies (separate totals)","All currencies"] else ""
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Net order revenue*", money(revenue,ccy))
    c2.metric("Orders", f"{n_orders:,}")
    c3.metric("Customers", f"{n_customers:,}")
    c4.metric("Average order value", money(aov,ccy))
    c5.metric("Return events", f"{returns_n:,}")
    if currency_choice in ["All currencies (separate totals)","All currencies"] and len(currency_values)>1:
        split=f_orders.groupby("currency")["order_total"].sum()
        st.caption("Currency totals are kept separate: " + " · ".join(f"{k}: {v:,.2f}" for k,v in split.items()))
    st.caption("*Excludes orders marked Cancelled or Returned. Currency values are not converted.")

def chart(fig, height=360):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo":False})

def show_table(df, height=380):
    if df is None or df.empty:
        st.info("No records match the current filters.")
    else:
        st.dataframe(df, use_container_width=True, height=height, hide_index=True)

def section(title, subtitle=None):
    st.markdown(f"### {title}")
    if subtitle: st.caption(subtitle)

def no_data():
    st.warning("No records match the selected filters. Clear or widen the filters to see data.")

# Header
st.markdown('<span class="rp-pill">DATA CONNECTED · CSV DATASET</span>', unsafe_allow_html=True)
st.title(page)
st.caption(f"{start_date:%d %b %Y} — {end_date:%d %b %Y}  |  {len(f_orders):,} filtered orders  |  {len(f_items):,} purchase lines")
if customer_search and f_orders.empty:
    st.error(f"No orders found for exact Customer ID `{customer_search}` in the selected date/country/channel filters. No unfiltered fallback is applied.")

if page == "Executive Summary":
    metric_row()
    if f_orders.empty: no_data()
    else:
        left,right=st.columns([1.55,1])
        daily=f_orders.groupby(f_orders["order_date"].dt.to_period("M")).agg(revenue=("order_total","sum"),orders=("order_id","nunique")).reset_index()
        daily["month"]=daily["order_date"].astype(str)
        with left:
            section("Revenue & order momentum","Monthly trend from filtered orders")
            fig=px.area(daily,x="month",y="revenue",markers=True)
            chart(fig,350)
        with right:
            section("Order status mix")
            status=f_orders["order_status"].value_counts().rename_axis("status").reset_index(name="orders")
            chart(px.pie(status,names="status",values="orders",hole=.62),350)
        a,b,c=st.columns(3)
        with a:
            section("Top categories")
            cat=f_items.groupby("category")["line_total"].sum().nlargest(7).reset_index(name="revenue")
            chart(px.bar(cat,x="revenue",y="category",orientation="h"),300)
        with b:
            section("Sales channel")
            ch=f_orders.groupby("sales_channel")["order_total"].sum().reset_index()
            chart(px.pie(ch,names="sales_channel",values="order_total",hole=.55),300)
        with c:
            section("Top countries")
            co=f_orders.groupby("shipping_country")["order_total"].sum().nlargest(7).reset_index()
            chart(px.bar(co,x="order_total",y="shipping_country",orientation="h"),300)
        section("Latest orders")
        show_table(f_orders.sort_values("order_datetime",ascending=False)[["order_id","invoice_id","customer_id","order_datetime","sales_channel","order_status","currency","order_total"]].head(20))

elif page == "Sales Trend Analysis":
    metric_row()
    if f_orders.empty: no_data()
    else:
        freq=st.selectbox("Trend granularity",["Daily","Weekly","Monthly"],index=2)
        tmp=f_orders.copy()
        tmp["period"]=tmp["order_date"].dt.to_period({"Daily":"D","Weekly":"W","Monthly":"M"}[freq]).astype(str)
        trend=tmp.groupby("period").agg(revenue=("order_total","sum"),orders=("order_id","nunique"),customers=("customer_id","nunique")).reset_index()
        x,y=st.columns(2)
        with x: chart(px.line(trend,x="period",y="revenue",markers=True,title="Order revenue"))
        with y: chart(px.bar(trend,x="period",y="orders",title="Order volume"))
        section("Trend data"); show_table(trend)

elif page == "Product Performance":
    metric_row()
    if f_items.empty: no_data()
    else:
        p=f_items.groupby(["product_id","product_name","category","sku"]).agg(units=("quantity","sum"),orders=("order_id","nunique"),revenue=("line_total","sum"),discount=("discount_amount","sum"),estimated_cost=("line_cost","sum")).reset_index()
        p["estimated_profit"]=p["line_subtotal"].groupby(p["product_id"]).transform("sum") if "line_subtotal" in p else 0
        p["estimated_profit"]=p["revenue"]-p["estimated_cost"]-p["discount"]
        a,b=st.columns(2)
        with a: chart(px.bar(p.nlargest(15,"revenue"),x="revenue",y="product_name",orientation="h",title="Top 15 products by revenue"))
        with b: chart(px.bar(p.groupby("category",as_index=False).agg(revenue=("revenue","sum")),x="category",y="revenue",title="Revenue by category"))
        section("Product detail"); show_table(p.sort_values("revenue",ascending=False).head(100))

elif page == "Customer Analytics":
    metric_row()
    if f_customers.empty: no_data()
    else:
        ca=f_customers.merge(cust_analytics.drop(columns=[c for c in f_customers.columns if c in cust_analytics.columns and c!="customer_id"],errors="ignore"),on="customer_id",how="left") if not cust_analytics.empty else f_customers
        # Recompute filter-period metrics to keep figures aligned with global filters.
        agg=f_orders.groupby("customer_id").agg(order_count=("order_id","nunique"),period_spend=("order_total","sum"),avg_order_value=("order_total","mean"),last_purchase=("order_date","max")).reset_index()
        ca=f_customers.merge(agg,on="customer_id",how="left")
        a,b=st.columns(2)
        with a: chart(px.histogram(ca,x="loyalty_tier",title="Customers by loyalty tier"))
        with b: chart(px.scatter(ca,x="order_count",y="period_spend",color="loyalty_tier",hover_data=["customer_id","customer_name"],title="Customer value map"))
        d1,d2,d3=st.columns(3)
        with d1:
            if "gender" in ca: chart(px.pie(ca,names="gender",hole=.5,title="Customer gender mix"),300)
        with d2:
            if "age" in ca:
                age_data=ca.copy(); age_data["age_group"]=pd.cut(age_data["age"],bins=[0,20,30,40,50,60,120],labels=["Under 20","20–29","30–39","40–49","50–59","60+"])
                chart(px.histogram(age_data,x="age_group",title="Customer age groups"),300)
        with d3:
            if "country" in ca: chart(px.bar(ca["country"].value_counts().head(10).rename_axis("country").reset_index(name="customers"),x="country",y="customers",title="Customer geography"),300)
        section("Customer profiles and filtered purchase metrics")
        show_table(ca.sort_values("period_spend",ascending=False),450)

elif page == "Region & Country Sales":
    metric_row()
    if f_orders.empty: no_data()
    else:
        region=f_orders.groupby(["shipping_country","currency"]).agg(orders=("order_id","nunique"),customers=("customer_id","nunique"),revenue=("order_total","sum")).reset_index()
        a,b=st.columns(2)
        with a: chart(px.bar(region,x="shipping_country",y="revenue",color="currency",barmode="group",title="Revenue by country (currency kept separate)"))
        with b: chart(px.treemap(region,path=["shipping_country"],values="orders",title="Order distribution"))
        section("Country performance"); show_table(region.sort_values("orders",ascending=False))

elif page == "Monthly & Seasonal Sales":
    metric_row()
    if f_orders.empty: no_data()
    else:
        m=f_orders.copy(); m["month_name"]=m["order_date"].dt.month_name(); m["month_num"]=m["order_date"].dt.month
        monthly=m.groupby(["order_year","order_month"]).agg(revenue=("order_total","sum"),orders=("order_id","nunique")).reset_index()
        monthly["period"]=monthly["order_year"].astype(str)+"-"+monthly["order_month"].astype(str).str.zfill(2)
        season=m.groupby("month_num").agg(revenue=("order_total","sum"),orders=("order_id","nunique")).reset_index()
        a,b=st.columns(2)
        with a: chart(px.line(monthly,x="period",y="revenue",markers=True,title="Monthly revenue"))
        with b: chart(px.bar(season,x="month_num",y="revenue",title="Seasonality by calendar month"))
        section("Monthly data"); show_table(monthly)

elif page == "Customer Behaviour":
    metric_row()
    if f_orders.empty: no_data()
    else:
        counts=f_orders.groupby("customer_id").agg(orders=("order_id","nunique"),spend=("order_total","sum"),last_purchase=("order_date","max")).reset_index()
        counts["recency_days"]=(pd.Timestamp(end_date)-counts["last_purchase"]).dt.days
        counts=counts.merge(customers[["customer_id","customer_name","loyalty_tier","country"]],on="customer_id",how="left")
        a,b=st.columns(2)
        with a: chart(px.histogram(counts,x="orders",nbins=20,title="Purchase frequency distribution"))
        with b: chart(px.box(counts,x="loyalty_tier",y="spend",title="Spend by loyalty tier"))
        section("Customer frequency and recency"); show_table(counts.sort_values("spend",ascending=False).head(500))

elif page == "Revenue & Profit Analysis":
    metric_row()
    if f_items.empty: no_data()
    else:
        # Estimate is derived from synthetic unit_cost in the supplied dataset, not audited accounting.
        f_items["estimated_profit"]=f_items["line_subtotal"]-f_items["discount_amount"]-f_items["line_cost"]
        total_rev=f_items["line_total"].sum(); estimated_profit=f_items["estimated_profit"].sum()
        a,b,c=st.columns(3); a.metric("Item revenue",money(total_rev)); b.metric("Estimated product profit",money(estimated_profit)); c.metric("Estimated profit rate",f"{estimated_profit/max(total_rev,1)*100:.1f}%")
        st.caption("Profit is an estimate using generated unit_cost values. It is not audited financial profit. Revenue and profit are grouped by each order's currency where applicable.")
        p=f_items.groupby("category").agg(revenue=("line_total","sum"),estimated_profit=("estimated_profit","sum")).reset_index()
        chart(px.bar(p,x="category",y=["revenue","estimated_profit"],barmode="group",title="Revenue vs estimated profit by category"))
        show_table(p)

elif page == "Inventory Risk":
    st.caption("Inventory values are a generated snapshot, not a live warehouse connection.")
    if f_inventory.empty: no_data()
    else:
        a,b,c=st.columns(3)
        a.metric("SKUs tracked",f"{f_inventory.product_id.nunique():,}")
        b.metric("Out of stock",f"{(f_inventory.stock_status=='Out of Stock').sum():,}")
        c.metric("Reorder required",f"{(f_inventory.stock_status=='Reorder').sum():,}")
        risk=f_inventory.copy()
        if selected_countries and "country" in risk: risk=risk[risk.country.isin(selected_countries)]
        chart(px.histogram(risk,x="stock_status",color="stock_status",title="Inventory health"))
        risk["stock_gap"]=(risk["reorder_level"]-risk["stock_on_hand"]).clip(lower=0)
        section("Items requiring attention")
        show_table(risk[risk.stock_status!="Healthy"].sort_values("stock_gap",ascending=False)[["product_id","sku","product_name","category","supplier_id","stock_on_hand","reorder_level","reorder_quantity","days_of_cover","stock_status"]])

elif page == "Orders & Transactions":
    metric_row()
    tabs=st.tabs(["Orders","Purchase lines","Payments","Shipments","Returns"])
    with tabs[0]:
        show_table(f_orders.sort_values("order_datetime",ascending=False),500)
    with tabs[1]:
        show_table(f_items.sort_values("order_id"),500)
    with tabs[2]:
        if f_payments.empty: no_data()
        else:
            p1,p2=st.columns(2)
            with p1:
                if "payment_method" in f_payments: chart(px.histogram(f_payments,x="payment_method",color="payment_status" if "payment_status" in f_payments else None,title="Payment methods and status"),320)
            with p2:
                if "payment_status" in f_payments: chart(px.pie(f_payments,names="payment_status",hole=.55,title="Payment status mix"),320)
            if "currency" in f_payments and "amount" in f_payments and "payment_method" in f_payments:
                pm=f_payments.groupby(["currency","payment_method"],as_index=False)["amount"].sum()
                chart(px.bar(pm,x="payment_method",y="amount",color="currency",barmode="group",title="Payment amount by method (currency separated)"),320)
            show_table(f_payments.sort_values("payment_datetime",ascending=False) if "payment_datetime" in f_payments else f_payments,400)
    with tabs[3]:
        if f_shipments.empty: no_data()
        else:
            s1,s2=st.columns(2)
            with s1: chart(px.histogram(f_shipments,x="shipping_status",title="Shipment status distribution"),320)
            with s2: chart(px.histogram(f_shipments,x="carrier",color="shipping_status",title="Shipments by carrier"),320)
            show_table(f_shipments.sort_values("estimated_delivery_date",ascending=False) if "estimated_delivery_date" in f_shipments else f_shipments,400)
    with tabs[4]:
        if f_returns.empty: no_data()
        else:
            r1,r2=st.columns(2)
            with r1: chart(px.histogram(f_returns,x="return_reason",title="Return reasons"),320)
            with r2: chart(px.histogram(f_returns,x="return_status",title="Return status"),320)
            if "refund_amount" in f_returns and "return_reason" in f_returns:
                refunds=f_returns.groupby("return_reason",as_index=False)["refund_amount"].sum()
                chart(px.bar(refunds,x="return_reason",y="refund_amount",title="Refund amount by return reason"),320)
            show_table(f_returns.sort_values("return_id",ascending=False) if "return_id" in f_returns else f_returns,400)

elif page == "Advanced Analytics":
    metric_row()
    if f_orders.empty: no_data()
    else:
        cust=f_orders.groupby("customer_id").agg(orders=("order_id","nunique"),spend=("order_total","sum"),last=("order_date","max")).reset_index()
        cust["recency_days"]=(pd.Timestamp(end_date)-cust["last"]).dt.days
        cust["frequency_band"]=pd.cut(cust["orders"],bins=[0,1,3,6,100000],labels=["1 order","2–3 orders","4–6 orders","7+ orders"],include_lowest=True)
        a,b=st.columns(2)
        with a: chart(px.scatter(cust,x="recency_days",y="spend",size="orders",color="frequency_band",title="Recency × frequency × spend"))
        with b: chart(px.histogram(cust,x="frequency_band",title="Customer frequency segments"))
        section("High-value customer opportunities")
        show_table(cust.sort_values("spend",ascending=False).head(100))
        st.markdown("### Campaign performance (filtered orders)")
        if "campaign_id" in f_orders and not campaigns.empty:
            camp_orders=f_orders[f_orders["campaign_id"].notna()].groupby("campaign_id").agg(orders=("order_id","nunique"),revenue=("order_total","sum"),customers=("customer_id","nunique")).reset_index()
            camp_view=camp_orders.merge(campaigns,on="campaign_id",how="left")
            if not camp_view.empty:
                c1,c2=st.columns(2)
                with c1: chart(px.bar(camp_view,x="campaign_name",y="orders",title="Orders by campaign"),330)
                with c2:
                    camp_view["revenue_per_budget_unit"]=camp_view["revenue"]/camp_view["budget"].replace(0,np.nan)
                    chart(px.bar(camp_view,x="campaign_name",y="revenue",title="Attributed order revenue by campaign"),330)
                show_table(camp_view.sort_values("revenue",ascending=False))
            else: st.info("No campaign-attributed orders match the current filters.")
        else: st.info("Campaign attribution is unavailable in the filtered data.")
        st.caption("Campaign attribution is simulated demo data; revenue is attributed by campaign_id on each order and is not proof of causal campaign lift.")

elif page == "Interactive Filters":
    metric_row()
    st.markdown("### Active filter state")
    st.json({"start_date":str(start_date),"end_date":str(end_date),"countries":selected_countries or "All","channels":selected_channels or "All","exact_customer_id":customer_search or "All","currency_view":currency_choice})
    st.markdown("### Filtered dataset preview")
    a,b=st.columns(2)
    with a:
        section("Filtered order records"); show_table(f_orders.head(100))
    with b:
        section("Filtered purchase line records"); show_table(f_items.head(100))
    st.download_button("Download filtered orders CSV",f_orders.to_csv(index=False).encode("utf-8"),"retailpulse_filtered_orders.csv","text/csv")
    st.download_button("Download filtered purchase lines CSV",f_items.to_csv(index=False).encode("utf-8"),"retailpulse_filtered_order_items.csv","text/csv")


elif page == "AI & Predictive Analytics":

    def load_ml_csv(relative_path):
        file_path = PROJECT_ROOT / relative_path
        if not file_path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(file_path, low_memory=False)
        except Exception as error:
            st.warning(f"Could not load {relative_path}: {error}")
            return pd.DataFrame()

    segmentation = load_ml_csv(
        "models/outputs/final_segmentation/final_customer_segments.csv"
    )
    segment_summary = load_ml_csv(
        "models/outputs/final_segmentation/final_segment_profiles.csv"
    )
    forecast_predictions = load_ml_csv(
        "models/outputs/forecasting/weekly_model_comparison_predictions.csv"
    )
    forecast_metrics = load_ml_csv(
        "models/outputs/forecasting/weekly_model_comparison.csv"
    )
    churn_predictions = load_ml_csv(
        "models/outputs/churn/churn_predictions.csv"
    )
    inventory_results = load_ml_csv(
        "models/outputs/inventory/inventory_optimization_results.csv"
    )
    reorder_recommendations = load_ml_csv(
        "models/outputs/inventory/reorder_recommendations.csv"
    )

    ai_tabs = st.tabs([
        "Customer Segmentation",
        "Demand Forecasting",
        "Churn Analytics",
        "Inventory Optimization",
    ])

    # --------------------------------------------------------
    # CUSTOMER SEGMENTATION
    # --------------------------------------------------------
    with ai_tabs[0]:
        section(
            "Customer Segmentation",
            "Business-oriented customer groups generated using K-Means clustering."
        )

        if segmentation.empty:
            st.warning("Final customer segmentation results were not found.")
        else:
            total_customers = len(segmentation)
            purchased = int(
                (segmentation["cluster"] != -1).sum()
            )
            no_purchase = total_customers - purchased
            total_segments = segmentation.loc[
                segmentation["cluster"] != -1, "cluster"
            ].nunique()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Customers", f"{total_customers:,}")
            m2.metric("Customers with Purchases", f"{purchased:,}")
            m3.metric("No Purchase Customers", f"{no_purchase:,}")
            m4.metric("Customer Segments", total_segments)

            section("Customer Segment Distribution")

            segment_counts = (
                segmentation["customer_segment"]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("customer_segment")
                .reset_index(name="customers")
            )

            a, b = st.columns(2)

            with a:
                chart(
                    px.pie(
                        segment_counts,
                        names="customer_segment",
                        values="customers",
                        hole=0.55,
                        title="Customer Segment Distribution"
                    ),
                    360
                )

            with b:
                chart(
                    px.bar(
                        segment_counts,
                        x="customer_segment",
                        y="customers",
                        title="Customers by Segment",
                        text="customers"
                    ),
                    360
                )

            if not segment_summary.empty:
                section("Customer Segment Profiles")

                display_profiles = segment_summary.rename(
                    columns={
                        "customer_segment": "Segment",
                        "customer_count": "Customers",
                        "average_recency": "Avg Recency (Days)",
                        "average_frequency": "Avg Orders",
                        "average_spending": "Avg Spending",
                        "average_order_value": "Avg Order Value",
                        "customer_percentage": "Customer Share (%)"
                    }
                )

                show_table(display_profiles, 300)

                section("Segment Spending Comparison")

                chart(
                    px.bar(
                        segment_summary,
                        x="customer_segment",
                        y="average_spending",
                        title="Average Customer Spending by Segment",
                        text_auto=".2f"
                    ),
                    360
                )

                section("Purchase Frequency Analysis")

                chart(
                    px.bar(
                        segment_summary,
                        x="customer_segment",
                        y="average_frequency",
                        title="Average Orders per Customer",
                        text_auto=".2f"
                    ),
                    360
                )

                section("Customer Recency Analysis")

                chart(
                    px.bar(
                        segment_summary,
                        x="customer_segment",
                        y="average_recency",
                        title="Average Days Since Last Purchase",
                        text_auto=".1f"
                    ),
                    360
                )

            section("Customer-Level Segmentation Results")

            selected_segments = st.multiselect(
                "Filter Customer Segments",
                sorted(segmentation["customer_segment"].dropna().unique()),
                default=sorted(segmentation["customer_segment"].dropna().unique()),
                key="final_customer_segment_filter"
            )

            filtered_segments = segmentation[
                segmentation["customer_segment"].isin(selected_segments)
            ]

            show_table(filtered_segments, 420)

            csv_data = filtered_segments.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Customer Segmentation CSV",
                data=csv_data,
                file_name="retailpulse_customer_segments.csv",
                mime="text/csv"
            )

            st.caption(
                "Model: K-Means | Clusters: 6 | "
                "Silhouette Score: 0.2636"
            )

    # --------------------------------------------------------
    # DEMAND FORECASTING
    # --------------------------------------------------------
    with ai_tabs[1]:
        section(
            "Weekly Demand Forecasting",
            "Actual versus predicted weekly revenue. INR and USD are displayed separately."
        )

        if forecast_predictions.empty:
            st.warning(
                "Weekly forecast predictions were not found. Expected "
                "models/outputs/forecasting/weekly_model_comparison_predictions.csv."
            )
        else:
            forecast_view = forecast_predictions.copy()

            if "Date" in forecast_view.columns:
                forecast_view["Date"] = pd.to_datetime(
                    forecast_view["Date"], errors="coerce"
                )

            if "Currency" in forecast_view.columns:
                available_currencies = sorted(
                    forecast_view["Currency"].dropna().astype(str).unique()
                )
            else:
                available_currencies = []

            if "Model" in forecast_view.columns:
                available_models = sorted(
                    forecast_view["Model"].dropna().astype(str).unique()
                )
            else:
                available_models = []

            f1, f2 = st.columns(2)

            with f1:
                selected_forecast_currency = st.selectbox(
                    "Forecast currency",
                    available_currencies,
                    key="ai_forecast_currency"
                ) if available_currencies else None

            with f2:
                preferred_model = (
                    "Seasonal Naive"
                    if selected_forecast_currency == "INR"
                    else "Prophet"
                )
                default_model_index = (
                    available_models.index(preferred_model)
                    if preferred_model in available_models
                    else 0
                )
                selected_forecast_model = st.selectbox(
                    "Forecast model",
                    available_models,
                    index=default_model_index,
                    key="ai_forecast_model"
                ) if available_models else None

            if (
                selected_forecast_currency is not None
                and selected_forecast_model is not None
            ):
                selected_forecast = forecast_view[
                    (forecast_view["Currency"].astype(str)
                     == selected_forecast_currency)
                    & (forecast_view["Model"].astype(str)
                       == selected_forecast_model)
                ].copy()

                if not selected_forecast.empty:
                    plot_data = selected_forecast.melt(
                        id_vars=["Date"],
                        value_vars=["Actual", "Predicted"],
                        var_name="Series",
                        value_name="Revenue"
                    )
                    chart(
                        px.line(
                            plot_data,
                            x="Date",
                            y="Revenue",
                            color="Series",
                            markers=True,
                            title=(
                                f"{selected_forecast_currency} weekly revenue: "
                                f"Actual vs {selected_forecast_model}"
                            )
                        ),
                        390
                    )
                    st.caption(
                        "The chart shows the saved eight-week holdout predictions. "
                        "It is a historical model evaluation, not a live future forecast."
                    )
                    section("Weekly prediction values")
                    show_table(
                        selected_forecast.sort_values("Date"),
                        330
                    )
                else:
                    st.info("No forecast rows match the selected currency and model.")

            if not forecast_metrics.empty:
                section("Forecast model comparison")
                show_table(forecast_metrics, 300)

                        # --------------------------------------------------------
        # 30-DAY FUTURE REVENUE FORECAST
        # --------------------------------------------------------
        st.divider()

        section(
            "30-Day Revenue Forecast",
            "Future revenue projections from Prophet and LSTM models."
        )

        prophet_path = (
            PROJECT_ROOT
            / "models/outputs/forecasting/demand_forecast_30days.csv"
        )

        lstm_path = (
            PROJECT_ROOT
            / "models/outputs/forecasting/daily_lstm_forecast_30days.csv"
        )

        if prophet_path.exists() and lstm_path.exists():

            prophet_future = pd.read_csv(prophet_path)
            lstm_future = pd.read_csv(lstm_path)

            prophet_future["ds"] = pd.to_datetime(
                prophet_future["ds"], errors="coerce"
            )
            lstm_future["Date"] = pd.to_datetime(
                lstm_future["Date"], errors="coerce"
            )

            selected_currency = st.selectbox(
                "Select forecast currency",
                ["INR", "USD"],
                key="future_forecast_currency"
            )

            prophet_selected = prophet_future[
                prophet_future["currency"] == selected_currency
            ].copy()

            lstm_selected = lstm_future[
                lstm_future["Currency"] == selected_currency
            ].copy()

            prophet_selected = prophet_selected.rename(
                columns={
                    "ds": "Date",
                    "yhat": "Prophet Forecast",
                    "yhat_lower": "Lower Bound",
                    "yhat_upper": "Upper Bound"
                }
            )

            lstm_selected = lstm_selected.rename(
                columns={
                    "Predicted_Revenue": "LSTM Forecast"
                }
            )

            future_comparison = pd.merge(
                prophet_selected[
                    [
                        "Date",
                        "Prophet Forecast",
                        "Lower Bound",
                        "Upper Bound"
                    ]
                ],
                lstm_selected[["Date", "LSTM Forecast"]],
                on="Date",
                how="inner"
            ).sort_values("Date")

            if not future_comparison.empty:

                c1, c2 = st.columns(2)

                c1.metric(
                    "Prophet Average",
                    f"{future_comparison['Prophet Forecast'].mean():,.2f}"
                )

                c2.metric(
                    "LSTM Average",
                    f"{future_comparison['LSTM Forecast'].mean():,.2f}"
                )

                forecast_plot = future_comparison.melt(
                    id_vars=["Date"],
                    value_vars=[
                        "Prophet Forecast",
                        "LSTM Forecast"
                    ],
                    var_name="Model",
                    value_name="Predicted Revenue"
                )

                chart(
                    px.line(
                        forecast_plot,
                        x="Date",
                        y="Predicted Revenue",
                        color="Model",
                        markers=True,
                        title=f"{selected_currency} - 30-Day Revenue Forecast"
                    ),
                    420
                )

                section("Future Forecast Details")
                show_table(future_comparison, 400)

                st.download_button(
                    "Download Forecast CSV",
                    future_comparison.to_csv(index=False).encode("utf-8"),
                    file_name=f"RetailPulse_{selected_currency}_Forecast.csv",
                    mime="text/csv"
                )

                st.caption(
                    "Future projections from two separate models. "
                    "These are not actual sales or an ensemble forecast."
                )

            else:
                st.warning("No matching forecast dates were found.")

        else:
            st.warning("30-day forecast files are missing.")

    # --------------------------------------------------------
    # CHURN ANALYTICS
    # --------------------------------------------------------
    with ai_tabs[2]:
        section(
            "Customer Inactivity Risk",
            "Baseline classifier output. Inactivity is defined as 180 or more days since the last purchase."
        )

        churn_metrics_path = (
            PROJECT_ROOT / "models" / "outputs" / "churn" / "churn_metrics.json"
        )
        churn_metrics = {}
        if churn_metrics_path.exists():
            try:
                import json
                with churn_metrics_path.open("r", encoding="utf-8") as metrics_file:
                    churn_metrics = json.load(metrics_file)
            except Exception as error:
                st.warning(f"Could not read churn metrics: {error}")

        if churn_metrics:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{churn_metrics.get('accuracy', 0):.3f}")
            m2.metric("Precision", f"{churn_metrics.get('precision', 0):.3f}")
            m3.metric("Recall", f"{churn_metrics.get('recall', 0):.3f}")
            m4.metric("F1-score", f"{churn_metrics.get('f1_score', 0):.3f}")
            auc_value = churn_metrics.get("roc_auc")
            m5.metric(
                "ROC-AUC",
                f"{auc_value:.3f}" if auc_value is not None else "N/A"
            )

        if churn_predictions.empty:
            st.info(
                "Churn predictions were not found. Expected "
                "models/outputs/churn/churn_predictions.csv."
            )
        else:
            if "risk_category" in churn_predictions.columns:
                risk_counts = (
                    churn_predictions["risk_category"]
                    .fillna("Unknown")
                    .value_counts()
                    .rename_axis("risk_category")
                    .reset_index(name="customers")
                )
                chart(
                    px.bar(
                        risk_counts,
                        x="risk_category",
                        y="customers",
                        title="Predicted customer inactivity risk"
                    ),
                    320
                )

            if "churn_probability" in churn_predictions.columns:
                chart(
                    px.histogram(
                        churn_predictions,
                        x="churn_probability",
                        nbins=25,
                        title="Distribution of predicted inactivity probability"
                    ),
                    320
                )

            section("Customer-level churn predictions")
            show_table(
                churn_predictions.sort_values(
                    "churn_probability",
                    ascending=False
                ) if "churn_probability" in churn_predictions.columns
                else churn_predictions,
                420
            )

    # --------------------------------------------------------
    # INVENTORY OPTIMIZATION
    # --------------------------------------------------------
    with ai_tabs[3]:
        section(
            "Inventory Optimization",
            "Saved stock-risk classification and reorder recommendations."
        )

        if inventory_results.empty:
            st.warning(
                "Inventory optimization results were not found. Expected "
                "models/outputs/inventory/inventory_optimization_results.csv."
            )
        else:
            total_products = len(inventory_results)
            reorder_count = (
                int(inventory_results["reorder_required"].sum())
                if "reorder_required" in inventory_results.columns
                else len(reorder_recommendations)
            )
            estimated_cost = (
                float(inventory_results["estimated_reorder_cost"].sum())
                if "estimated_reorder_cost" in inventory_results.columns
                else 0.0
            )

            critical_count = (
                int((inventory_results["inventory_risk"] == "Critical").sum())
                if "inventory_risk" in inventory_results.columns
                else 0
            )

            i1, i2, i3, i4 = st.columns(4)
            i1.metric("Products tracked", f"{total_products:,}")
            i2.metric("Critical products", f"{critical_count:,}")
            i3.metric("Products to reorder", f"{reorder_count:,}")
            i4.metric("Estimated reorder cost", f"{estimated_cost:,.2f}")

            if "inventory_risk" in inventory_results.columns:
                risk_counts = (
                    inventory_results["inventory_risk"]
                    .value_counts()
                    .rename_axis("inventory_risk")
                    .reset_index(name="products")
                )
                chart(
                    px.bar(
                        risk_counts,
                        x="inventory_risk",
                        y="products",
                        title="Inventory risk distribution"
                    ),
                    330
                )

            if not reorder_recommendations.empty:
                section("Reorder recommendations")
                show_table(
                    reorder_recommendations.sort_values(
                        "days_of_cover"
                    ) if "days_of_cover" in reorder_recommendations.columns
                    else reorder_recommendations,
                    430
                )

                st.download_button(
                    "Download reorder recommendations",
                    reorder_recommendations.to_csv(index=False).encode("utf-8"),
                    "retailpulse_reorder_recommendations.csv",
                    "text/csv"
                )
            else:
                st.info("No reorder recommendation rows were found.")

            section("Full inventory optimization results")
            show_table(inventory_results, 430)
# --------------------------------------------------------
# MLFLOW MODEL PERFORMANCE & TRACKING
# --------------------------------------------------------

elif page == "Model Performance & Tracking":

    st.title("Model Performance & Tracking")

    st.markdown(
        "Monitor RetailPulse model evaluation and "
        "data processing performance."
    )

    output_dir = PROJECT_ROOT / "models" / "outputs"

    summary_file = output_dir / "performance_summary.json"
    results_file = output_dir / "performance_test_results.csv"

    if summary_file.exists() and results_file.exists():

        import json

        with open(summary_file, "r") as f:
            summary = json.load(f)

        results = pd.read_csv(results_file)

        st.subheader("Data Processing Performance")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Files Processed",
            summary["total_files_processed"]
        )

        c2.metric(
            "Processing Time",
            f'{summary["total_processing_time_seconds"]:.3f} sec'
        )

        c3.metric(
            "Target Time",
            f'{summary["target_seconds"]} sec'
        )

        c4.metric(
            "Target Status",
            "Achieved" if summary["target_met"] else "Not Achieved"
        )

        st.markdown("---")

        st.subheader("File Processing Results")

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Processing Time Comparison")

        fig = px.bar(
            results,
            x="File",
            y="Processing_Time_Seconds",
            color="Status",
            title="Processing Time by Dataset",
            text_auto=".3f"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Dataset Processing Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Total Rows Processed",
            f'{results["Rows"].sum():,}'
        )

        c2.metric(
            "Total Columns",
            f'{results["Columns"].sum():,}'
        )

        c3.metric(
            "Successful Files",
            f'{(results["Status"] == "SUCCESS").sum()}'
        )

        st.download_button(
            "Download Performance Results",
            results.to_csv(index=False).encode("utf-8"),
            "retailpulse_performance_results.csv",
            "text/csv"
        )

        st.caption(
            "Results represent the saved CSV processing benchmark. "
            "They are not machine learning accuracy metrics."
        )

    else:
        st.warning(
            "Performance result files are not available."
        )
st.markdown("---")
st.caption("RetailPulse · Dataset-aligned analytics | Synthetic demonstration data. Simulated records are not real customer transactions.")
