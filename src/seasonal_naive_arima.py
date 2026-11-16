
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load dataset
df = pd.read_csv("data/raw/train.csv")
df["date"] = pd.to_datetime(df["date"])

# 2. Select Store 1, Item 1
series = df[
    (df["store"] == 1) &
    (df["item"] == 1)
].copy()

series = series.sort_values("date")
series = series.set_index("date")["sales"]
series = series.asfreq("D")

# 3. Fixed-origin train-test split
train = series.loc[:"2016-12-31"]
test = series.loc["2017-01-01":"2017-12-31"]

# 4. Seasonal Naive forecast
# Repeat the last seven observed training values
last_week = train.iloc[-7:].values

predictions = np.tile(
    last_week,
    int(np.ceil(len(test) / 7))
)[:len(test)]

# 5. Evaluate
mae = mean_absolute_error(test, predictions)

rmse = np.sqrt(
    mean_squared_error(test, predictions)
)

wape = (
    np.abs(test.values - predictions).sum()
    / test.sum()
) * 100

print("\n========== FIXED-ORIGIN SEASONAL NAIVE ==========")

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"WAPE: {wape:.2f}%")

print("\nTraining observations:", len(train))
print("Testing observations:", len(test))
