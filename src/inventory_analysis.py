
import pandas as pd
import numpy as np
import os

# 1. Load historical sales data
df = pd.read_csv("data/raw/train.csv")

df["date"] = pd.to_datetime(df["date"])

# 2. Define inventory assumptions
LEAD_TIME_DAYS = 7
SERVICE_LEVEL = 0.95
Z_SCORE = 1.645

# 3. Calculate demand statistics for every store-item pair
inventory = df.groupby(["store", "item"]).agg(
    average_daily_demand=("sales", "mean"),
    daily_demand_std=("sales", "std"),
    total_sales=("sales", "sum")
).reset_index()

# 4. Calculate safety stock
inventory["safety_stock"] = (
    Z_SCORE
    * inventory["daily_demand_std"]
    * np.sqrt(LEAD_TIME_DAYS)
)

# 5. Calculate lead-time demand
inventory["lead_time_demand"] = (
    inventory["average_daily_demand"]
    * LEAD_TIME_DAYS
)

# 6. Calculate reorder point
inventory["reorder_point"] = (
    inventory["lead_time_demand"]
    + inventory["safety_stock"]
)

# 7. Round inventory quantities upward
inventory["safety_stock"] = np.ceil(
    inventory["safety_stock"]
).astype(int)

inventory["reorder_point"] = np.ceil(
    inventory["reorder_point"]
).astype(int)

# 8. Display results
print("\n========== INVENTORY ANALYSIS ==========")

print("\nTotal store-item combinations:", len(inventory))

print("\nSample Inventory Recommendations:")

print(
    inventory.head(10).to_string(index=False)
)

# 9. Save results
os.makedirs("data/processed", exist_ok=True)

inventory.to_csv(
    "data/processed/inventory_recommendations.csv",
    index=False
)

print("\nInventory analysis completed successfully!")
print("Results saved to data/processed/inventory_recommendations.csv")
