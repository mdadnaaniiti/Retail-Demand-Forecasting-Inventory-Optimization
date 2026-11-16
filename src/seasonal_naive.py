
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
df = pd.read_csv(BASE_DIR / "data" / "raw" / "train.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["store", "item", "date"]
).copy()

# Chronological split
train = df[df["date"].dt.year <= 2016].copy()
validation = df[df["date"].dt.year == 2017].copy()

# Create seasonal naive predictions
# Same weekday from the previous week
validation["predicted_sales"] = (
    validation.groupby(["store", "item"])["sales"]
    .shift(7)
)

# Use the final 7 training days for the first validation week
last_7_days = (
    train.groupby(["store", "item"])
    .tail(7)
)

history = last_7_days[
    ["date", "store", "item", "sales"]
].copy()

history["date"] = history["date"] + pd.Timedelta(days=7)

history = history.rename(
    columns={"sales": "predicted_sales"}
)

# Match shifted historical sales to validation dates
validation = validation.merge(
    history[["date", "store", "item", "predicted_sales"]],
    on=["date", "store", "item"],
    how="left",
    suffixes=("", "_history")
)

validation["predicted_sales"] = (
    validation["predicted_sales"]
    .fillna(validation["predicted_sales_history"])
)

validation = validation.dropna(
    subset=["predicted_sales"]
)

# Actual and predicted values
y_true = validation["sales"]
y_pred = validation["predicted_sales"]

# Evaluation
mae = mean_absolute_error(y_true, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_true, y_pred)
)

wape = (
    np.abs(y_true - y_pred).sum()
    / np.abs(y_true).sum()
) * 100

# Display results
print("\n========== SEASONAL NAIVE RESULTS ==========")

print("Training records:", len(train))
print("Validation records:", len(validation))

print("\nEvaluation Metrics:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"WAPE: {wape:.2f}%")

print("\nSeasonal naive model completed successfully!")
