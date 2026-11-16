
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("data/raw/train.csv")

df["date"] = pd.to_datetime(df["date"])

# Sort data
df = df.sort_values(["store", "item", "date"])

# Calculate rolling 7-day demand for each store-item combination
df["7day_demand"] = (
    df.groupby(["store", "item"])["sales"]
    .transform(lambda x: x.rolling(window=7).sum())
)

# Remove incomplete initial windows
df = df.dropna(subset=["7day_demand"])

# Calculate inventory statistics
inventory_stats = (
    df.groupby(["store", "item"])["7day_demand"]
    .agg(
        average_7day_demand="mean",
        std_7day_demand="std",
        reorder_point_95=lambda x: x.quantile(0.95)
    )
    .reset_index()
)

# Round inventory quantities upward
inventory_stats["average_7day_demand"] = np.ceil(
    inventory_stats["average_7day_demand"]
)

inventory_stats["std_7day_demand"] = np.ceil(
    inventory_stats["std_7day_demand"]
)

inventory_stats["reorder_point_95"] = np.ceil(
    inventory_stats["reorder_point_95"]
)

# Save results
inventory_stats.to_csv(
    "data/processed/lead_time_demand.csv",
    index=False
)

print("7-day demand analysis completed!")
print("Total store-item combinations:", len(inventory_stats))

print("\nSample results:")
print(inventory_stats.head())

print("\nSaved to data/processed/lead_time_demand.csv")
