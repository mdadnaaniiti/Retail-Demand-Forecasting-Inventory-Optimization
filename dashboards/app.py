
import streamlit as st
import pandas as pd
import altair as alt
from pathlib import Path

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw"
PROCESSED_PATH = BASE_DIR / "data" / "processed"

# ==========================================
# 2. LOAD DATA
# ==========================================

@st.cache_data
def load_sales_data():
    df = pd.read_csv(RAW_PATH / "train.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_forecast_data():
    forecasts = pd.read_csv(
        PROCESSED_PATH / "future_7day_forecasts.csv"
    )
    summary = pd.read_csv(
        PROCESSED_PATH / "future_7day_demand_summary.csv"
    )

    forecasts["target_date"] = pd.to_datetime(
        forecasts["target_date"]
    )
    forecasts["forecast_origin"] = pd.to_datetime(
        forecasts["forecast_origin"]
    )

    return forecasts, summary


df = load_sales_data()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("📊 Retail Analytics")

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Sales Analysis",
        "Demand Forecasting",
        "Inventory Optimization",
        "Model Backtesting"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Retail Demand Forecasting Project")

# ==========================================
# 4. OVERVIEW PAGE
# ==========================================

if page == "Overview":

    st.title("📊 Retail Demand Forecasting & Inventory Optimization")

    st.markdown(
        "An end-to-end machine learning application for "
        "retail demand forecasting and inventory planning "
        "across multiple stores and products."
    )

    st.divider()

    total_sales = df["sales"].sum()
    total_stores = df["store"].nunique()
    total_items = df["item"].nunique()
    total_records = len(df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Sales", f"{total_sales:,.0f}")
    col2.metric("Total Stores", total_stores)
    col3.metric("Total Products", total_items)
    col4.metric("Total Records", f"{total_records:,}")

    st.divider()

    st.subheader("Overall Sales Trend")

    daily_sales = (
        df.groupby("date")["sales"]
        .sum()
        .reset_index()
    )

    st.line_chart(
        daily_sales.set_index("date")["sales"],
        use_container_width=True
    )

    st.subheader("Store-wise Sales Performance")

    store_sales = (
        df.groupby("store", as_index=False)["sales"]
        .sum()
        .rename(columns={"sales": "total_sales"})
        .sort_values("store")
    )

    st.bar_chart(
        store_sales,
        x="store",
        y="total_sales",
        x_label="Store",
        y_label="Total Sales",
        use_container_width=True
    )

    st.subheader("Monthly Demand Seasonality")

    monthly_sales = (
        df.groupby(df["date"].dt.month)["sales"]
        .mean()
        .reset_index()
    )

    monthly_sales.columns = [
        "month", "average_daily_sales"
    ]

    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar",
        4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep",
        10: "Oct", 11: "Nov", 12: "Dec"
    }

    monthly_sales["month"] = (
        monthly_sales["month"].map(month_names)
    )

    month_order = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    monthly_chart = alt.Chart(monthly_sales).mark_bar().encode(
        x=alt.X("month:N", sort=month_order, title="Month"),
        y=alt.Y(
            "average_daily_sales:Q",
            title="Average Daily Sales"
        ),
        tooltip=["month", "average_daily_sales"]
    ).properties(height=350)

    st.altair_chart(
        monthly_chart,
        use_container_width=True
    )

# ==========================================
# 5. DEMAND FORECASTING PAGE
# ==========================================

elif page == "Demand Forecasting":

    st.title("📈 Seven-Day Demand Forecasting")

    st.markdown(
        "Explore seven-day demand predictions generated "
        "using our trained multi-horizon XGBoost model."
    )

    forecasts, summary = load_forecast_data()

    st.divider()

    # Store and item selection

    col1, col2 = st.columns(2)

    with col1:
        selected_store = st.selectbox(
            "Select Store",
            sorted(forecasts["store"].unique())
        )

    with col2:
        available_items = sorted(
            forecasts.loc[
                forecasts["store"] == selected_store,
                "item"
            ].unique()
        )

        selected_item = st.selectbox(
            "Select Item",
            available_items
        )

    # Filter daily forecasts

    selected_forecast = forecasts[
        (forecasts["store"] == selected_store) &
        (forecasts["item"] == selected_item)
    ].sort_values("target_date")

    selected_summary = summary[
        (summary["store"] == selected_store) &
        (summary["item"] == selected_item)
    ]

    forecast_origin = selected_forecast[
        "forecast_origin"
    ].iloc[0]

    seven_day_demand = selected_summary[
        "forecast_7day_demand"
    ].iloc[0]

    st.divider()

    # Forecast KPIs

    col1, col2, col3 = st.columns(3)

    col1.metric("Selected Store", selected_store)
    col2.metric("Selected Item", selected_item)
    col3.metric(
        "Predicted 7-Day Demand",
        f"{seven_day_demand:,.0f} units"
    )

    st.caption(
        f"Forecast origin: {forecast_origin.strftime('%d %b %Y')}"
    )

    st.divider()

    # Daily forecast chart

    st.subheader("Daily Demand Forecast")

    forecast_chart = alt.Chart(
        selected_forecast
    ).mark_line(
        point=True
    ).encode(
        x=alt.X("target_date:T", title="Forecast Date"),
        y=alt.Y("predicted_sales:Q", title="Predicted Sales"),
        tooltip=[
            "target_date:T",
            "horizon:Q",
            "predicted_sales:Q"
        ]
    ).properties(height=400)

    st.altair_chart(
        forecast_chart,
        use_container_width=True
    )

    # Forecast table

    st.subheader("Forecast Details")

    display_forecast = selected_forecast[
        ["target_date", "horizon", "predicted_sales"]
    ].copy()

    display_forecast.columns = [
        "Forecast Date",
        "Forecast Horizon (Days)",
        "Predicted Sales"
    ]

    st.dataframe(
        display_forecast,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "This forecast is generated from the dataset's final "
        "historical date (31 December 2017). It is a project "
        "demonstration, not a live forecast for today's date."
    )

# ==========================================
# 6. OTHER PAGES - TO BE BUILT
# ==========================================


elif page == "Sales Analysis":

    st.title("📊 Sales Analysis")

    st.markdown(
        "Explore historical sales patterns across stores "
        "and products from 2013 to 2017."
    )

    st.divider()

    # ======================================
    # FILTERS
    # ======================================

    col1, col2 = st.columns(2)

    with col1:
        selected_store = st.selectbox(
            "Select Store",
            ["All Stores"] + sorted(df["store"].unique().tolist()),
            key="sales_store"
        )

    with col2:
        selected_item = st.selectbox(
            "Select Product",
            ["All Products"] + sorted(df["item"].unique().tolist()),
            key="sales_item"
        )

    filtered_df = df.copy()

    if selected_store != "All Stores":
        filtered_df = filtered_df[
            filtered_df["store"] == selected_store
        ]

    if selected_item != "All Products":
        filtered_df = filtered_df[
            filtered_df["item"] == selected_item
        ]

    st.divider()

    # ======================================
    # SALES KPIs
    # ======================================

    total_sales = filtered_df["sales"].sum()

    daily_sales = (
        filtered_df.groupby("date")["sales"].sum()
    )

    average_daily_sales = daily_sales.mean()

    highest_daily_sales = daily_sales.max()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", f"{total_sales:,.0f}")
    col2.metric("Average Daily Sales", f"{average_daily_sales:,.2f}")
    col3.metric("Highest Daily Sales", f"{highest_daily_sales:,.0f}")

    st.divider()

    # ======================================
    # DAILY SALES TREND
    # ======================================

    st.subheader("Daily Sales Trend")

    trend = (
        filtered_df.groupby("date")["sales"]
        .sum()
        .reset_index()
    )

    trend_chart = alt.Chart(trend).mark_line().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("sales:Q", title="Sales"),
        tooltip=["date:T", "sales:Q"]
    ).properties(height=350)

    st.altair_chart(
        trend_chart,
        use_container_width=True
    )

    # ======================================
    # PRODUCT PERFORMANCE
    # ======================================

    st.subheader("Top 10 Products by Sales")

    item_sales = (
        filtered_df.groupby("item")["sales"]
        .sum()
        .reset_index()
        .sort_values("sales", ascending=False)
        .head(10)
    )

    item_chart = alt.Chart(item_sales).mark_bar().encode(
        x=alt.X("item:O", title="Product"),
        y=alt.Y("sales:Q", title="Total Sales"),
        tooltip=["item", "sales"]
    ).properties(height=350)

    st.altair_chart(
        item_chart,
        use_container_width=True
    )

    # ======================================
    # WEEKDAY ANALYSIS
    # ======================================

    st.subheader("Day-of-Week Demand Pattern")

    weekday_sales = filtered_df.copy()

    weekday_sales["weekday"] = (
        weekday_sales["date"].dt.day_of_week
    )

    weekday_avg = (
        weekday_sales.groupby("weekday")["sales"]
        .mean()
        .reset_index()
    )

    weekday_names = {
        0: "Monday", 1: "Tuesday", 2: "Wednesday",
        3: "Thursday", 4: "Friday",
        5: "Saturday", 6: "Sunday"
    }

    weekday_avg["day"] = weekday_avg["weekday"].map(
        weekday_names
    )

    weekday_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    weekday_chart = alt.Chart(weekday_avg).mark_bar().encode(
        x=alt.X("day:N", sort=weekday_order, title="Day"),
        y=alt.Y("sales:Q", title="Average Daily Sales"),
        tooltip=["day", "sales"]
    ).properties(height=350)

    st.altair_chart(
        weekday_chart,
        use_container_width=True
    )


elif page == "Inventory Optimization":

    st.title("📦 Inventory Optimization")

    st.markdown(
        "Forecast-driven inventory planning with reorder "
        "recommendations and stockout-risk monitoring."
    )

    # Load inventory recommendations
    @st.cache_data
    def load_inventory_data():
        path = PROCESSED_PATH / "forecast_based_inventory.csv"
        return pd.read_csv(path)

    inventory = load_inventory_data()

    # Identify status and risk columns
    status_col = next(
        (c for c in ["status", "inventory_status"]
         if c in inventory.columns),
        None
    )

    risk_col = next(
        (c for c in ["stockout_risk", "risk_level", "risk_category"]
         if c in inventory.columns),
        None
    )

    if status_col is None or risk_col is None:
        st.error(
            "Could not identify the inventory status or risk columns. "
            "Please check the CSV column names."
        )
        st.write("Available columns:", inventory.columns.tolist())

    else:

        st.divider()

        # KPI calculations
        total_combinations = len(inventory)

        reorder_count = (
            inventory[status_col] == "Reorder Now"
        ).sum()

        sufficient_count = (
    inventory[status_col] != "Reorder Now"
).sum()
        total_replenishment = inventory[
            "replenishment_quantity"
        ].sum()

        # KPI cards
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Store-Item Combinations",
            f"{total_combinations:,}"
        )

        col2.metric(
            "Reorder Required",
            f"{reorder_count:,}"
        )

        col3.metric(
            "Sufficient Stock",
            f"{sufficient_count:,}"
        )

        col4.metric(
            "Total Replenishment",
            f"{total_replenishment:,.0f} units"
        )

        st.divider()

        # Filters
        st.subheader("Inventory Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            selected_store = st.selectbox(
                "Select Store",
                ["All Stores"] +
                sorted(inventory["store"].unique().tolist())
            )

        with col2:
            selected_status = st.selectbox(
                "Filter by Inventory Status",
                ["All", "Reorder Now", "Sufficient"]
            )

        filtered = inventory.copy()

        if selected_store != "All Stores":
            filtered = filtered[
                filtered["store"] == selected_store
            ]

        if selected_status != "All":
            filtered = filtered[
                filtered[status_col] == selected_status
            ]

        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # Reorder status chart
        st.subheader("Inventory Reorder Status")

        status_counts = (
            inventory[status_col]
            .value_counts()
            .rename_axis("status")
            .reset_index(name="count")
        )

        status_chart = alt.Chart(
            status_counts
        ).mark_bar().encode(
            x=alt.X("status:N", title="Inventory Status"),
            y=alt.Y("count:Q", title="Number of Items"),
            tooltip=["status", "count"]
        ).properties(height=350)

        st.altair_chart(
            status_chart,
            use_container_width=True
        )

        # Stockout risk analysis
        st.subheader("Stockout Risk Distribution")

        risk_counts = (
            inventory[risk_col]
            .value_counts()
            .rename_axis("risk")
            .reset_index(name="count")
        )

        risk_chart = alt.Chart(
            risk_counts
        ).mark_bar().encode(
            x=alt.X("risk:N", title="Stockout Risk"),
            y=alt.Y("count:Q", title="Number of Items"),
            tooltip=["risk", "count"]
        ).properties(height=350)

        st.altair_chart(
            risk_chart,
            use_container_width=True
        )

        st.info(
            "Inventory levels are simulated because the original "
            "dataset does not contain actual stock or purchase-order "
            "records. Reorder recommendations use our forecast-based "
            "inventory policy."
        )


elif page == "Model Backtesting":

    st.title("🎯 Model Backtesting")

    st.markdown(
        "Comparison of historical-demand-based and "
        "forecast-based inventory policies."
    )

    @st.cache_data
    def load_backtest_data():
        return pd.read_csv(
            PROCESSED_PATH / "inventory_policy_summary.csv"
        )

    results = load_backtest_data()

    historical = results[
        results["policy"].str.lower() == "historical"
    ].iloc[0]

    forecast = results[
        results["policy"].str.lower() == "forecast"
    ].iloc[0]

    st.divider()

    # KPI calculations
    service_improvement = (
        forecast["service_level_pct"]
        - historical["service_level_pct"]
    )

    unmet_reduction = (
        (historical["unmet_demand"] - forecast["unmet_demand"])
        / historical["unmet_demand"]
    ) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Forecast Service Level",
        f'{forecast["service_level_pct"]:.2f}%'
    )

    col2.metric(
        "Service Level Improvement",
        f"+{service_improvement:.2f} percentage points"
    )

    col3.metric(
        "Reduction in Unmet Demand",
        f"{unmet_reduction:.2f}%"
    )

    st.divider()

    # Prepare comparison data
    comparison = pd.DataFrame({
        "Policy": ["Historical", "Forecast-based"],
        "Service Level (%)": [
            historical["service_level_pct"],
            forecast["service_level_pct"]
        ],
        "Unmet Demand": [
            historical["unmet_demand"],
            forecast["unmet_demand"]
        ],
        "Stockout Days": [
            historical["stockout_days"],
            forecast["stockout_days"]
        ],
        "Average Inventory": [
            historical["average_inventory"],
            forecast["average_inventory"]
        ]
    })

    st.subheader("Service Level Comparison")

    service_chart = alt.Chart(
        comparison
    ).mark_bar().encode(
        x=alt.X("Policy:N", title="Inventory Policy"),
        y=alt.Y(
            "Service Level (%):Q",
            title="Service Level (%)",
            scale=alt.Scale(domain=[0, 100])
        ),
        tooltip=["Policy", "Service Level (%)"]
    ).properties(height=350)

    st.altair_chart(
        service_chart,
        use_container_width=True
    )

    st.subheader("Unmet Demand Comparison")

    unmet_chart = alt.Chart(
        comparison
    ).mark_bar().encode(
        x=alt.X("Policy:N", title="Inventory Policy"),
        y=alt.Y("Unmet Demand:Q", title="Unmet Demand (Units)"),
        tooltip=["Policy", "Unmet Demand"]
    ).properties(height=350)

    st.altair_chart(
        unmet_chart,
        use_container_width=True
    )

    st.subheader("Detailed Policy Comparison")

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "These results come from a simulated 2017 inventory "
        "backtest. The simulation assumes a 7-day lead time, "
        "weekly review, and lost sales when demand cannot be met. "
        "They should not be interpreted as actual retailer performance."
    )

# ==========================================
# 7. FOOTER
# ==========================================

st.divider()

st.caption(
    "Retail Demand Forecasting & Inventory Optimization | "
    "Python | XGBoost | Streamlit"
)
