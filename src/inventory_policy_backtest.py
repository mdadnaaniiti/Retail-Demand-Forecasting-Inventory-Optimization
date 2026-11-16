
import pandas as pd
import numpy as np


# ==================================================
# 1. LOAD DATA
# ==================================================

sales = pd.read_csv("data/raw/train.csv")
sales["date"] = pd.to_datetime(sales["date"])

weekly_forecasts = pd.read_csv(
    "data/processed/inventory_backtest_weekly.csv"
)
weekly_forecasts["date"] = pd.to_datetime(
    weekly_forecasts["date"]
)

# Use only forecast origins whose 7-day window
# belongs to the 2017 validation period.
weekly_forecasts = weekly_forecasts[
    (weekly_forecasts["date"] >= "2016-12-31") &
    (weekly_forecasts["date"] <= "2017-12-24")
].copy()

# Actual daily demand during 2017
sales_2017 = sales[
    (sales["date"] >= "2017-01-01") &
    (sales["date"] <= "2017-12-31")
].copy()


# ==================================================
# 2. ESTIMATE HISTORICAL INVENTORY PARAMETERS
#    USING TRAINING DATA ONLY (2013-2016)
# ==================================================

training_sales = sales[
    sales["date"] < "2017-01-01"
].copy()

training_sales = training_sales.sort_values(
    ["store", "item", "date"]
)

training_sales["rolling_7day_demand"] = (
    training_sales.groupby(
        ["store", "item"]
    )["sales"]
    .transform(lambda x: x.rolling(7).sum())
)

historical_stats = (
    training_sales
    .dropna(subset=["rolling_7day_demand"])
    .groupby(["store", "item"])["rolling_7day_demand"]
    .agg(
        average_7day_demand="mean",
        reorder_point="quantile"
    )
    .reset_index()
)

# Calculate the 95th percentile explicitly
percentile_stats = (
    training_sales
    .dropna(subset=["rolling_7day_demand"])
    .groupby(["store", "item"])["rolling_7day_demand"]
    .quantile(0.95)
    .reset_index(name="reorder_point")
)

historical_stats = historical_stats.drop(
    columns=["reorder_point"]
).merge(
    percentile_stats,
    on=["store", "item"]
)

historical_stats["average_7day_demand"] = np.ceil(
    historical_stats["average_7day_demand"]
)

historical_stats["reorder_point"] = np.ceil(
    historical_stats["reorder_point"]
)

historical_stats["safety_stock"] = (
    historical_stats["reorder_point"] -
    historical_stats["average_7day_demand"]
).clip(lower=0)


# ==================================================
# 3. PREPARE DAILY ACTUAL DEMAND
# ==================================================

actual_lookup = {
    (int(row.store), int(row.item), row.date): int(row.sales)
    for row in sales_2017.itertuples(index=False)
}

stats_lookup = historical_stats.set_index(
    ["store", "item"]
).to_dict("index")


# ==================================================
# 4. INVENTORY SIMULATION FUNCTION
# ==================================================

def simulate_policy(policy_name):

    daily_records = []
    order_records = []

    for (store, item), group in weekly_forecasts.groupby(
        ["store", "item"]
    ):

        key = (int(store), int(item))

        stats = stats_lookup[key]

        avg_weekly_demand = stats["average_7day_demand"]
        historical_rop = stats["reorder_point"]
        safety_stock = stats["safety_stock"]

        # Same initial stock for both policies
        on_hand = int(
            historical_rop + avg_weekly_demand
        )

        # Orders waiting to arrive
        pipeline = {}

        group = group.sort_values("date")

        for row in group.itertuples(index=False):

            origin = row.date

            # Forecast-based or historical reorder point
            if policy_name == "Historical":

                reorder_point = historical_rop

            else:

                reorder_point = (
                    np.ceil(row.predicted_7day_demand)
                    + safety_stock
                )

            # Order-up-to target includes another
            # 7 days of average demand (review period)
            target_inventory = (
                reorder_point + avg_weekly_demand
            )

            # Inventory position includes outstanding orders
            inventory_position = (
                on_hand + sum(pipeline.values())
            )

            order_quantity = 0

            if inventory_position <= reorder_point:

                order_quantity = max(
                    0,
                    int(np.ceil(
                        target_inventory - inventory_position
                    ))
                )

                if order_quantity > 0:

                    arrival_date = (
                        origin + pd.Timedelta(days=7)
                    )

                    pipeline[arrival_date] = (
                        pipeline.get(arrival_date, 0)
                        + order_quantity
                    )

                    order_records.append({
                        "policy": policy_name,
                        "store": store,
                        "item": item,
                        "order_date": origin,
                        "arrival_date": arrival_date,
                        "order_quantity": order_quantity
                    })

            # Simulate actual demand during the next 7 days
            for day_offset in range(1, 8):

                current_date = (
                    origin +
                    pd.Timedelta(days=day_offset)
                )

                if not (
                    pd.Timestamp("2017-01-01")
                    <= current_date
                    <= pd.Timestamp("2017-12-31")
                ):
                    continue

                # Receive orders before daily demand
                if current_date in pipeline:

                    on_hand += pipeline.pop(current_date)

                demand = actual_lookup.get(
                    (int(store), int(item), current_date),
                    0
                )

                fulfilled = min(on_hand, demand)

                unmet_demand = demand - fulfilled

                on_hand -= fulfilled

                daily_records.append({
                    "policy": policy_name,
                    "date": current_date,
                    "store": store,
                    "item": item,
                    "actual_demand": demand,
                    "fulfilled_demand": fulfilled,
                    "unmet_demand": unmet_demand,
                    "ending_inventory": on_hand,
                    "stockout_day": int(unmet_demand > 0)
                })

    return (
        pd.DataFrame(daily_records),
        pd.DataFrame(order_records)
    )


# ==================================================
# 5. RUN BOTH POLICIES
# ==================================================

print("Running historical inventory simulation...")

historical_daily, historical_orders = simulate_policy(
    "Historical"
)

print("Running forecast-based inventory simulation...")

forecast_daily, forecast_orders = simulate_policy(
    "Forecast"
)

daily_results = pd.concat(
    [historical_daily, forecast_daily],
    ignore_index=True
)

order_results = pd.concat(
    [historical_orders, forecast_orders],
    ignore_index=True
)


# ==================================================
# 6. CALCULATE PERFORMANCE METRICS
# ==================================================

def calculate_metrics(group):

    total_demand = group["actual_demand"].sum()

    total_unmet = group["unmet_demand"].sum()

    total_fulfilled = group["fulfilled_demand"].sum()

    service_level = (
        total_fulfilled / total_demand * 100
        if total_demand > 0 else 0
    )

    return pd.Series({
        "total_demand": total_demand,
        "fulfilled_demand": total_fulfilled,
        "unmet_demand": total_unmet,
        "stockout_days": group["stockout_day"].sum(),
        "average_inventory": group["ending_inventory"].mean(),
        "service_level_pct": service_level
    })


summary = (
    daily_results.groupby("policy")
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
)

order_summary = (
    order_results.groupby("policy")
    .agg(
        total_orders=("order_quantity", "count"),
        total_order_quantity=("order_quantity", "sum")
    )
    .reset_index()
)

summary = summary.merge(
    order_summary,
    on="policy",
    how="left"
).fillna({
    "total_orders": 0,
    "total_order_quantity": 0
})


# ==================================================
# 7. DISPLAY RESULTS
# ==================================================

print("\n========== INVENTORY POLICY BACKTEST ==========")

print(summary.to_string(index=False))

print("\n========== STORE-WISE PERFORMANCE ==========")

store_summary = (
    daily_results.groupby(["policy", "store"])
    .apply(calculate_metrics, include_groups=False)
    .reset_index()
)

print(store_summary.to_string(index=False))


# ==================================================
# 8. SAVE RESULTS
# ==================================================

daily_results.to_csv(
    "data/processed/inventory_policy_daily_results.csv",
    index=False
)

order_results.to_csv(
    "data/processed/inventory_policy_orders.csv",
    index=False
)

summary.to_csv(
    "data/processed/inventory_policy_summary.csv",
    index=False
)

store_summary.to_csv(
    "data/processed/inventory_policy_store_summary.csv",
    index=False
)

print("\nInventory policy backtesting completed!")
print("All results saved in data/processed/")
