
import pandas as pd
import numpy as np
import warnings

from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# 1. Load data
df = pd.read_csv("data/raw/train.csv")
df["date"] = pd.to_datetime(df["date"])

# 2. Select Store 1, Item 1
series = df[
    (df["store"] == 1) &
    (df["item"] == 1)
].copy()

series = series.sort_values("date")
series = series.set_index("date")["sales"].asfreq("D")

# 3. Train-validation-test split
train = series.loc[:"2015-12-31"]
valid = series.loc["2016-01-01":"2016-12-31"]

print("Training observations:", len(train))
print("Validation observations:", len(valid))

# 4. Define parameter combinations
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]

P_values = [0, 1]
D_values = [0, 1]
Q_values = [0, 1]

parameter_combinations = list(
    product(
        p_values, d_values, q_values,
        P_values, D_values, Q_values
    )
)

results = []

print("\nStarting SARIMA parameter tuning...")

# 5. Grid search
for i, (p, d, q, P, D, Q) in enumerate(
    parameter_combinations, start=1
):

    try:
        model = SARIMAX(
            train,
            order=(p, d, q),
            seasonal_order=(P, D, Q, 7),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted_model = model.fit(
            disp=False,
            maxiter=100
        )

        # Forecast the 2016 validation period
        predictions = fitted_model.forecast(
            steps=len(valid)
        )

        mae = mean_absolute_error(valid, predictions)

        rmse = np.sqrt(
            mean_squared_error(valid, predictions)
        )

        wape = (
            np.abs(valid.values - predictions.values).sum()
            / valid.sum()
        ) * 100

        results.append({
            "order": (p, d, q),
            "seasonal_order": (P, D, Q, 7),
            "MAE": mae,
            "RMSE": rmse,
            "WAPE": wape
        })

        print(
            f"{i}/{len(parameter_combinations)} completed | "
            f"SARIMA({p},{d},{q})x({P},{D},{Q},7) | "
            f"WAPE: {wape:.2f}%"
        )

    except Exception as e:
        print(f"Combination {i} failed: {e}")

# 6. Display and save results
results_df = pd.DataFrame(results)

if results_df.empty:
    raise RuntimeError("No SARIMA configuration completed successfully.")

results_df = results_df.sort_values("WAPE")

print("\n========== TOP 10 SARIMA CONFIGURATIONS ==========")
print(results_df.head(10).to_string(index=False))

results_df.to_csv(
    "data/processed/sarima_tuning_results.csv",
    index=False
)

print("\nTuning completed!")
print("Results saved successfully!")
