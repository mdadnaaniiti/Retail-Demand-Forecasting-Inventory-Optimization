
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load dataset
df = pd.read_csv("data/processed/sales_features.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["store", "item", "date"]
).copy()

# 2. Create seven-day target
df["target_sales"] = (
    df.groupby(["store", "item"])["sales"].shift(-7)
)

df["target_date"] = (
    df["date"] + pd.Timedelta(days=7)
)

# 3. Match the validation period used by XGBoost
feature_columns = [
    "store", "item",
    "year", "month", "day", "day_of_week",
    "day_of_year", "weekend",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14",
    "rolling_mean_30"
]

df = df.dropna(
    subset=feature_columns + ["target_sales"]
).copy()

valid = df[
    (df["target_date"] >= "2017-01-01") &
    (df["target_date"] <= "2017-12-31")
].copy()

# 4. Seasonal Naive prediction
# Today's observed sales predict sales seven days later.
valid["prediction"] = valid["sales"]

actual = valid["target_sales"]
predicted = valid["prediction"]

# 5. Calculate metrics
mae = mean_absolute_error(actual, predicted)

rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

wape = (
    np.abs(actual - predicted).sum()
    / actual.sum()
) * 100

print("\n========== 7-DAY SEASONAL NAIVE RESULTS ==========")

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"WAPE: {wape:.2f}%")

print("\nValidation records:", len(valid))
