
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load raw dataset
df = pd.read_csv(BASE_DIR / "data" / "raw" / "train.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Sort chronologically within each store-item combination
df = df.sort_values(["store", "item", "date"]).copy()

# Calendar features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.day_of_week
df["day_of_year"] = df["date"].dt.day_of_year
df["weekend"] = (df["day_of_week"] >= 5).astype(int)

# Group by store and item
grouped = df.groupby(["store", "item"])["sales"]

# Lag features
for lag in [1, 7, 14, 28]:
    df[f"lag_{lag}"] = grouped.shift(lag)

# Rolling features
# Shift by one day to exclude current day's sales
for window in [7, 14, 30]:
    df[f"rolling_mean_{window}"] = (
        df.groupby(["store", "item"])["sales"]
        .transform(
            lambda x: x.shift(1).rolling(window).mean()
        )
    )

# Save processed features
output_path = BASE_DIR / "data" / "processed"
output_path.mkdir(parents=True, exist_ok=True)

file_path = output_path / "sales_features.csv"
df.to_csv(file_path, index=False)

print("\n========== FEATURE ENGINEERING ==========")
print("Dataset shape:", df.shape)

print("\nCreated features:")
print(df.columns.tolist())

print("\nMissing values in engineered features:")
feature_columns = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14",
    "rolling_mean_30"
]

print(df[feature_columns].isnull().sum())

print("\nSaved file:", file_path)
print("\nFeature engineering completed successfully!")
