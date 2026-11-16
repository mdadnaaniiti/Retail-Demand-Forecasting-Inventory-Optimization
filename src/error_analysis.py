
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# 1. Load dataset and trained model
df = pd.read_csv("data/processed/sales_features.csv")
df["date"] = pd.to_datetime(df["date"])

model = joblib.load("models/xgboost_model.pkl")

# 2. Define the same features used during training
feature_columns = [
    "store", "item",
    "year", "month", "day", "day_of_week",
    "day_of_year", "weekend",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30"
]

# 3. Prepare validation data
df = df.dropna(subset=feature_columns + ["sales"]).copy()

valid = df[df["date"] >= "2017-01-01"].copy()

X_valid = valid[feature_columns]
y_valid = valid["sales"]

# 4. Generate predictions
valid["predicted_sales"] = model.predict(X_valid)

# Calculate absolute errors
valid["absolute_error"] = abs(
    valid["sales"] - valid["predicted_sales"]
)

# 5. Store-wise error analysis
store_results = valid.groupby("store").apply(
    lambda group: pd.Series({
        "MAE": group["absolute_error"].mean(),
        "WAPE": (
            group["absolute_error"].sum()
            / group["sales"].sum()
        ) * 100
    }),
    include_groups=False
).reset_index()

store_results = store_results.sort_values("WAPE")

# 6. Item-wise error analysis
item_results = valid.groupby("item").apply(
    lambda group: pd.Series({
        "MAE": group["absolute_error"].mean(),
        "WAPE": (
            group["absolute_error"].sum()
            / group["sales"].sum()
        ) * 100
    }),
    include_groups=False
).reset_index()

item_results = item_results.sort_values("WAPE")

# 7. Display results
print("\n========== STORE-WISE ERROR ANALYSIS ==========")
print(store_results.to_string(index=False))

print("\n========== ITEM-WISE ERROR ANALYSIS ==========")
print(item_results.to_string(index=False))

# 8. Save results
os.makedirs("data/processed", exist_ok=True)

store_results.to_csv(
    "data/processed/store_error_analysis.csv",
    index=False
)

item_results.to_csv(
    "data/processed/item_error_analysis.csv",
    index=False
)

# 9. Visualize store-wise WAPE
plt.figure(figsize=(10, 5))
plt.bar(store_results["store"].astype(str), store_results["WAPE"])
plt.xlabel("Store")
plt.ylabel("WAPE (%)")
plt.title("XGBoost Forecasting Error by Store")
plt.tight_layout()
plt.savefig("data/processed/store_error_analysis.png", dpi=300)
plt.show()

# 10. Visualize item-wise WAPE
plt.figure(figsize=(12, 6))
plt.bar(item_results["item"].astype(str), item_results["WAPE"])
plt.xlabel("Item")
plt.ylabel("WAPE (%)")
plt.title("XGBoost Forecasting Error by Item")
plt.tight_layout()
plt.savefig("data/processed/item_error_analysis.png", dpi=300)
plt.show()

print("\nError analysis completed successfully!")
