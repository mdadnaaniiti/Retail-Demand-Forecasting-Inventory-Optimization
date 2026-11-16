
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

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

# 3. Train-test split
train = series.loc[:"2016-12-31"]
test = series.loc["2017-01-01":"2017-12-31"]

print("Training observations:", len(train))
print("Testing observations:", len(test))

# 4. Train ARIMA model
print("\nTraining ARIMA model...")

arima_model = ARIMA(
    train,
    order=(1, 0, 1)
)

arima_fit = arima_model.fit()

arima_predictions = arima_fit.forecast(
    steps=len(test)
)

# 5. Train SARIMA model
print("\nTraining SARIMA model...")

sarima_model = SARIMAX(
    train,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
)

sarima_fit = sarima_model.fit(
    disp=False,
    maxiter=200
)

sarima_predictions = sarima_fit.forecast(
    steps=len(test)
)

# 6. Evaluation function
def evaluate_model(actual, predicted, model_name):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    wape = (
        np.abs(actual - predicted).sum()
        / actual.sum()
    ) * 100

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape
    }

# 7. Compare models
results = []

results.append(
    evaluate_model(test, arima_predictions, "ARIMA")
)

results.append(
    evaluate_model(test, sarima_predictions, "SARIMA")
)

results_df = pd.DataFrame(results)

print("\n========== ARIMA vs SARIMA RESULTS ==========")
print(results_df.to_string(index=False))

# 8. Save results
os.makedirs("data/processed", exist_ok=True)

results_df.to_csv(
    "data/processed/arima_sarima_comparison.csv",
    index=False
)

# 9. Plot actual vs predicted sales
plt.figure(figsize=(14, 6))

plt.plot(test.index, test.values, label="Actual Sales")
plt.plot(
    test.index,
    arima_predictions.values,
    label="ARIMA Forecast"
)

plt.plot(
    test.index,
    sarima_predictions.values,
    label="SARIMA Forecast"
)

plt.title("ARIMA vs SARIMA - Store 1, Item 1")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()

plt.savefig(
    "data/processed/arima_sarima_forecast.png",
    dpi=300
)

plt.show()

print("\nARIMA and SARIMA experiments completed!")
