
import pandas as pd
import numpy as np

# 1. Load both inventory files
historical = pd.read_csv(
    "data/processed/optimized_inventory.csv"
)

forecast = pd.read_csv(
    "data/processed/forecast_based_inventory.csv"
)

# 2. Select and rename historical columns
historical = historical[
    [
        "store",
        "item",
        "current_inventory",
        "reorder_point",
        "replenishment_quantity",
        "inventory_status"
    ]
].rename(
    columns={
        "reorder_point": "historical_reorder_point",
        "replenishment_quantity": "historical_replenishment",
        "inventory_status": "historical_status"
    }
)

# 3. Select and rename forecast columns
forecast = forecast[
    [
        "store",
        "item",
        "current_inventory",
        "stock_requirement",
        "replenishment_quantity",
        "status"
    ]
].rename(
    columns={
        "stock_requirement": "forecast_stock_requirement",
        "replenishment_quantity": "forecast_replenishment",
        "status": "forecast_status"
    }
)

# 4. Merge both methods
comparison = historical.merge(
    forecast,
    on=["store", "item"],
    how="inner",
    suffixes=("_historical", "_forecast")
)

# 5. Validate inventory consistency
if not (
    comparison["current_inventory_historical"] ==
    comparison["current_inventory_forecast"]
).all():
    raise ValueError("Current inventory values do not match!")

if len(comparison) != 500:
    raise ValueError(f"Expected 500 rows, got {len(comparison)}")

comparison["current_inventory"] = (
    comparison["current_inventory_historical"]
)

# 6. Compare decisions
comparison["decision_changed"] = (
    comparison["historical_status"] !=
    comparison["forecast_status"]
)

comparison["replenishment_difference"] = (
    comparison["forecast_replenishment"] -
    comparison["historical_replenishment"]
)

# 7. Overall summary
historical_reorders = (
    comparison["historical_status"] == "Reorder Now"
).sum()

forecast_reorders = (
    comparison["forecast_status"] == "Reorder Now"
).sum()

historical_quantity = (
    comparison["historical_replenishment"].sum()
)

forecast_quantity = (
    comparison["forecast_replenishment"].sum()
)

changed_decisions = comparison["decision_changed"].sum()

print("\n========== INVENTORY METHOD COMPARISON ==========")

print("\nTotal combinations:", len(comparison))

print("\nHistorical-based reorders:", historical_reorders)
print("Forecast-based reorders:", forecast_reorders)

print("\nHistorical replenishment quantity:", historical_quantity)
print("Forecast replenishment quantity:", forecast_quantity)

print("\nDifference in replenishment quantity:",
      forecast_quantity - historical_quantity)

print("\nChanged replenishment decisions:", changed_decisions)

# 8. Store-wise comparison
store_comparison = (
    comparison.groupby("store")
    .agg(
        historical_reorders=(
            "historical_status",
            lambda x: (x == "Reorder Now").sum()
        ),
        forecast_reorders=(
            "forecast_status",
            lambda x: (x == "Reorder Now").sum()
        ),
        historical_quantity=(
            "historical_replenishment",
            "sum"
        ),
        forecast_quantity=(
            "forecast_replenishment",
            "sum"
        )
    )
    .reset_index()
)

print("\n========== STORE-WISE COMPARISON ==========")
print(store_comparison.to_string(index=False))

# 9. Save results
comparison.to_csv(
    "data/processed/inventory_method_comparison.csv",
    index=False
)

store_comparison.to_csv(
    "data/processed/store_inventory_comparison.csv",
    index=False
)

print("\nComparison files saved successfully!")
