
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from statsmodels.tsa.statespace.sarimax import SARIMAX
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
series = series.set_index("date")["sales"].asfreq("D")

# 3. Train-test split
train = series.loc[:"2016-12-31"]
test = series.loc["2017-01-01":"2017-12-31"]

print("Training observations:", len(train))
print("Testing observations:", len(test))

# 4. Train selected SARIMA model
print("\nTraining tuned SARIMA model...")

model = SARIMAX(
    train,
    order=(1, 0, 2),
    seasonal_order=(1, 0, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

fitted_model = model.fit(
    disp=False,
    maxiter=200
)

# 5. Generate fixed-origin forecast
predictions = fitted_model.forecast(
    steps=len(test)
)

# 6. Evaluate model
mae = mean_absolute_error(test, predictions)

rmse = np.sqrt(
    mean_squared_error(test, predictions)
)

wape = (
    np.abs(test.values - predictions.values).sum()
    / test.sum()
) * 100

print("\n========== TUNED SARIMA FINAL RESULTS ==========")

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"WAPE: {wape:.2f}%")

# 7. Save predictions
os.makedirs("data/processed", exist_ok=True)

forecast_df = pd.DataFrame({
    "date": test.index,
    "actual_sales": test.values,
    "predicted_sales": predictions.values
})

forecast_df.to_csv(
    "data/processed/tuned_sarima_predictions.csv",
    index=False
)

# 8. Plot actual vs predicted
plt.figure(figsize=(14, 6))

plt.plot(test.index, test.values, label="Actual Sales")
plt.plot(
    test.index,
    predictions.values,
    label="Tuned SARIMA Forecast"
)

plt.title("Tuned SARIMA - Store 1, Item 1")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()

plt.savefig(
    "data/processed/tuned_sarima_forecast.png",
    dpi=300
)

plt.show()

print("\nFinal SARIMA evaluation completed!")