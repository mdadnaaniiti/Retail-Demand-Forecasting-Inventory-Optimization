
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load original dataset
df = pd.read_csv(BASE_DIR / "data" / "raw" / "train.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["store", "item", "date"]
).copy()

# Chronological split
train = df[df["date"].dt.year <= 2016].copy()
validation = df[df["date"].dt.year == 2017].copy()

# Naive forecast: previous day's sales
validation["predicted_sales"] = (
    validation.groupby(["store", "item"])["sales"]
    .shift(1)
)

# First validation day needs the last training day's sales
last_train_sales = (
    train.sort_values("date")
    .groupby(["store", "item"])["sales"]
    .last()
)

first_validation_date = validation["date"].min()

mask = validation["date"] == first_validation_date

validation.loc[mask, "predicted_sales"] = (
    validation.loc[mask]
    .set_index(["store", "item"])
    .index.map(last_train_sales)
)

# Remove rows without predictions
validation = validation.dropna(
    subset=["predicted_sales"]
)

y_true = validation["sales"]
y_pred = validation["predicted_sales"]

# Evaluation metrics
mae = mean_absolute_error(y_true, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_true, y_pred)
)

wape = (
    np.abs(y_true - y_pred).sum()
    / np.abs(y_true).sum()
) * 100

# Results
print("\n========== NAIVE BASELINE RESULTS ==========")

print("Training period:",
      train["date"].min().date(),
      "to",
      train["date"].max().date())

print("Validation period:",
      validation["date"].min().date(),
      "to",
      validation["date"].max().date())

print("Training records:", len(train))
print("Validation records:", len(validation))

print("\nEvaluation Metrics:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"WAPE: {wape:.2f}%")

print("\nBaseline model completed successfully!")
