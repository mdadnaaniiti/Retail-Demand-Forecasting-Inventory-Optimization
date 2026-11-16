
import pandas as pd
import numpy as np
import os

# 1. Load inventory recommendations
inventory = pd.read_csv(
    "data/processed/inventory_recommendations.csv"
)

# 2. Define simulation assumptions
LEAD_TIME_DAYS = 7
REVIEW_PERIOD_DAYS = 7
EXTRA_STOCK_DAYS = 7

# 3. Simulate current inventory
np.random.seed(42)

inventory["current_inventory"] = np.ceil(
    inventory["reorder_point"]
    * np.random.uniform(0.5, 1.5, len(inventory))
).astype(int)

# 4. Calculate target inventory level
inventory["target_inventory"] = (
    inventory["reorder_point"]
    + inventory["average_daily_demand"] * EXTRA_STOCK_DAYS
)

inventory["target_inventory"] = np.ceil(
    inventory["target_inventory"]
).astype(int)

# 5. Determine replenishment requirement
inventory["reorder_required"] = (
    inventory["current_inventory"]
    <= inventory["reorder_point"]
)

# 6. Calculate replenishment quantity
inventory["replenishment_quantity"] = np.where(
    inventory["reorder_required"],
    np.maximum(
        0,
        inventory["target_inventory"]
        - inventory["current_inventory"]
    ),
    0
)

# 7. Estimate days of inventory remaining
inventory["days_of_inventory"] = (
    inventory["current_inventory"]
    / inventory["average_daily_demand"]
)

# 8. Estimate stockout risk
inventory["stockout_risk"] = np.where(
    inventory["days_of_inventory"] < LEAD_TIME_DAYS,
    "High",
    np.where(
        inventory["days_of_inventory"] < LEAD_TIME_DAYS + REVIEW_PERIOD_DAYS,
        "Medium",
        "Low"
    )
)

# 9. Assign inventory status
inventory["inventory_status"] = np.where(
    inventory["current_inventory"] <= inventory["reorder_point"],
    "Reorder Now",
    "Sufficient Stock"
)

# 10. Display results
print("\n========== INVENTORY OPTIMIZATION RESULTS ==========")

print("\nTotal store-item combinations:", len(inventory))

print(
    "\nProducts requiring replenishment:",
    inventory["reorder_required"].sum()
)

print("\nStockout risk distribution:")
print(inventory["stockout_risk"].value_counts())

print("\nSample inventory recommendations:")

display_columns = [
    "store",
    "item",
    "current_inventory",
    "reorder_point",
    "target_inventory",
    "replenishment_quantity",
    "days_of_inventory",
    "stockout_risk",
    "inventory_status"
]

print(
    inventory[display_columns]
    .head(15)
    .to_string(index=False)
)

# 11. Save optimized inventory recommendations
os.makedirs("data/processed", exist_ok=True)

inventory.to_csv(
    "data/processed/optimized_inventory.csv",
    index=False
)

print("\nInventory optimization completed!")
print("Saved to data/processed/optimized_inventory.csv")
