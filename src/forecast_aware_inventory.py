
import pandas as pd
import numpy as np

# Load 7-day demand statistics
df = pd.read_csv("data/processed/lead_time_demand.csv")

# Assumptions
np.random.seed(42)

# Simulate current inventory
df["current_inventory"] = np.random.uniform(
    0.5, 1.5, len(df)
) * df["reorder_point_95"]

df["current_inventory"] = np.floor(
    df["current_inventory"]
).astype(int)

# Target inventory:
# Reorder point + one additional week of average demand
df["target_inventory"] = (
    df["reorder_point_95"] +
    df["average_7day_demand"]
)

df["target_inventory"] = np.ceil(
    df["target_inventory"]
).astype(int)

# Replenishment decision
df["reorder_required"] = (
    df["current_inventory"] <= df["reorder_point_95"]
)

df["replenishment_quantity"] = np.where(
    df["reorder_required"],
    df["target_inventory"] - df["current_inventory"],
    0
)

# Days of inventory remaining (approximate)
df["days_of_inventory"] = (
    df["current_inventory"] /
    (df["average_7day_demand"] / 7)
)

# Risk classification (heuristic)
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

# Save output
df.to_csv(
    "data/processed/forecast_aware_inventory.csv",
    index=False
)

# Summary
print("Inventory optimization completed!")

print("\nReplenishment summary:")
print(df["status"].value_counts())

print("\nStockout risk summary:")
print(df["stockout_risk"].value_counts())

print("\nSample recommendations:")
print(
    df[
        [
            "store",
            "item",
            "current_inventory",
            "reorder_point_95",
            "replenishment_quantity",
            "days_of_inventory",
            "stockout_risk",
            "status"
        ]
    ].head(10)
)

print("\nSaved to data/processed/forecast_aware_inventory.csv")
