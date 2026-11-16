
import pandas as pd
import matplotlib.pyplot as plt
import os

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. Load dataset
df = pd.read_csv("data/raw/train.csv")
df["date"] = pd.to_datetime(df["date"])

# 2. Select one store-item combination
series = df[
    (df["store"] == 1) &
    (df["item"] == 1)
].copy()

series = series.sort_values("date")
series = series.set_index("date")["sales"]

# Ensure daily frequency
series = series.asfreq("D")

print("Selected series: Store 1, Item 1")
print("Number of observations:", len(series))
print("Date range:", series.index.min(), "to", series.index.max())

# 3. Plot original time series
plt.figure(figsize=(12, 5))
plt.plot(series)
plt.title("Daily Sales - Store 1, Item 1")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("data/processed/arima_original_series.png", dpi=300)
plt.show()

# 4. Calculate rolling statistics
rolling_mean = series.rolling(window=30).mean()
rolling_std = series.rolling(window=30).std()

plt.figure(figsize=(12, 5))
plt.plot(series, label="Original Sales")
plt.plot(rolling_mean, label="30-day Rolling Mean")
plt.plot(rolling_std, label="30-day Rolling Std")
plt.title("Rolling Statistics - Store 1, Item 1")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig("data/processed/arima_rolling_statistics.png", dpi=300)
plt.show()

# 5. Augmented Dickey-Fuller test
adf_result = adfuller(series.dropna())

print("\n========== ADF TEST RESULTS ==========")
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Critical Values:", adf_result[4])

if adf_result[1] <= 0.05:
    print("Conclusion: Series is stationary at the 5% significance level.")
else:
    print("Conclusion: Series is non-stationary at the 5% significance level.")

# 6. ACF and PACF plots
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

plot_acf(series, lags=40, ax=axes[0])
axes[0].set_title("Autocorrelation Function (ACF)")

plot_pacf(series, lags=40, ax=axes[1], method="ywm")
axes[1].set_title("Partial Autocorrelation Function (PACF)")

plt.tight_layout()
plt.savefig("data/processed/arima_acf_pacf.png", dpi=300)
plt.show()

print("\nARIMA time-series analysis completed!")
