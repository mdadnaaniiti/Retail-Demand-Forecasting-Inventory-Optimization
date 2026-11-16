
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Load input files
# --------------------------------------------------

forecast = pd.read_csv(
    "data/processed/future_7day_demand_summary.csv"
)

historical = pd.read_csv(
    "data/processed/lead_time_demand.csv"
)

previous_inventory = pd.read_csv(
    "data/processed/optimized_inventory.csv"
)

# Keep the previously simulated inventory quantities
previous_inventory = previous_inventory[
    ["store", "item", "current_inventory"]
].copy()


# --------------------------------------------------
# 2. Merge all datasets
# --------------------------------------------------

df = forecast.merge(
    historical,
    on=["store", "item"],
    how="inner"
)

df = df.merge(
    previous_inventory,
    on=["store", "item"],
    how="inner"
)

if len(df) != 500:
    raise ValueError(
        f"Expected 500 combinations, found {len(df)}"
    )


# --------------------------------------------------
# 3. Calculate forecast-based safety buffer
# --------------------------------------------------

# Historical safety buffer:
# 95th percentile demand minus average 7-day demand

df["safety_stock"] = (
    df["reorder_point_95"] -
    df["average_7day_demand"]
).clip(lower=0)

# Round safety stock upward
df["safety_stock"] = np.ceil(
    df["safety_stock"]
)



# --------------------------------------------------
# 4. Calculate forecast-based stock requirements
# --------------------------------------------------

# Expected demand during the next 7 days
df["forecast_demand"] = np.ceil(
    df["forecast_7day_demand"]
)

# Forecast-based reorder point
df["forecast_reorder_point"] = (
    df["forecast_demand"] +
    df["safety_stock"]
)

# Target inventory includes another 7 days
# of average demand for the review period
df["target_inventory"] = (
    df["forecast_reorder_point"] +
    df["average_7day_demand"]
)

df["target_inventory"] = np.ceil(
    df["target_inventory"]
)

# Reorder decision
df["reorder_required"] = (
    df["current_inventory"] <=
    df["forecast_reorder_point"]
)

# Replenishment quantity
df["replenishment_quantity"] = np.where(
    df["reorder_required"],
    df["target_inventory"] -
    df["current_inventory"],
    0
)

# Stock requirement for reporting
df["stock_requirement"] = df["forecast_reorder_point"]



# --------------------------------------------------
# 5. Calculate estimated inventory coverage
# --------------------------------------------------

df["forecast_daily_demand"] = (
    df["forecast_7day_demand"] / 7
)

df["days_of_inventory"] = np.where(
    df["forecast_daily_demand"] > 0,
    df["current_inventory"] /
    df["forecast_daily_demand"],
    np.inf
)

df["stockout_risk"] = np.select(
    [
        df["days_of_inventory"] < 7,
        df["days_of_inventory"] < 14
    ],
    ["High", "Medium"],
    default="Low"
)

df["status"] = np.where(
    df["reorder_required"],
    "Reorder Now",
    "Sufficient Stock"
)


# --------------------------------------------------
# 6. Save recommendations
# --------------------------------------------------

output_columns = [
    "store",
    "item",
    "current_inventory",
    "forecast_7day_demand",
    "safety_stock",
    "stock_requirement",
    "replenishment_quantity",
    "days_of_inventory",
    "stockout_risk",
    "status"
]

df[output_columns].to_csv(
    "data/processed/forecast_based_inventory.csv",
    index=False
)


# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("\n========== FORECAST-BASED INVENTORY ==========")

print("\nTotal combinations:", len(df))

print("\nReplenishment summary:")
print(df["status"].value_counts())

print("\nStockout risk summary:")
print(df["stockout_risk"].value_counts())

print("\nTotal units recommended for replenishment:",
      int(df["replenishment_quantity"].sum()))

print("\nSample recommendations:")
print(df[output_columns].head(10))

print(
    "\nSaved to data/processed/forecast_based_inventory.csv"
)
