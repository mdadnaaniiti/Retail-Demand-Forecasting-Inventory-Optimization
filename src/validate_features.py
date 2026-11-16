
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    BASE_DIR / "data" / "processed" / "sales_features.csv"
)

df["date"] = pd.to_datetime(df["date"])

# Select one store-item combination
sample = df[
    (df["store"] == 1) &
    (df["item"] == 1)
].copy()

sample = sample.sort_values("date")

# Display first 35 observations
columns = [
    "date",
    "sales",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_30"
]

print("\n========== FEATURE VALIDATION ==========")
print(sample[columns].head(35).to_string(index=False))

# Validate lag_1
assert (
    sample["lag_1"].iloc[1] ==
    sample["sales"].iloc[0]
)

# Validate lag_7
assert (
    sample["lag_7"].iloc[7] ==
    sample["sales"].iloc[0]
)

# Validate rolling_mean_7
expected_mean = sample["sales"].iloc[0:7].mean()

actual_mean = sample["rolling_mean_7"].iloc[7]

assert abs(expected_mean - actual_mean) < 1e-9

print("\nAll feature validation checks passed!")
