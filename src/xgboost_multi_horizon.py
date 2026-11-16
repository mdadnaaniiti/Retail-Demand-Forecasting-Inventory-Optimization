
import pandas as pd
import numpy as np
import os
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Load feature-engineered dataset
# --------------------------------------------------

df = pd.read_csv("data/processed/sales_features.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["store", "item", "date"]
).copy()


# --------------------------------------------------
# 2. Define historical features
# --------------------------------------------------

base_features = [
    "store",
    "item",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30"
]


# --------------------------------------------------
# 3. Create training data for all 7 horizons
# --------------------------------------------------

forecast_data = []

for horizon in range(1, 8):

    temp = df.copy()

    # Sales on the future target date
    temp["target_sales"] = (
        temp.groupby(["store", "item"])["sales"]
        .shift(-horizon)
    )

    # Target date
    temp["target_date"] = (
        temp["date"] +
        pd.Timedelta(days=horizon)
    )

    # Calendar features of target date
    temp["target_year"] = temp["target_date"].dt.year
    temp["target_month"] = temp["target_date"].dt.month
    temp["target_day_of_week"] = (
        temp["target_date"].dt.day_of_week
    )
    temp["target_day_of_year"] = (
        temp["target_date"].dt.day_of_year
    )

    temp["target_weekend"] = (
        temp["target_day_of_week"] >= 5
    ).astype(int)

    temp["horizon"] = horizon

    # Keep one forecast origin every 7 days
    temp = temp[
        temp["date"].map(
            lambda x: x.toordinal() % 7 == 0
        )
    ]

    forecast_data.append(temp)


# Combine all horizons
forecast_df = pd.concat(
    forecast_data,
    ignore_index=True
)

# Remove missing values
feature_columns = (
    base_features +
    [
        "target_year",
        "target_month",
        "target_day_of_week",
        "target_day_of_year",
        "target_weekend",
        "horizon"
    ]
)

forecast_df = forecast_df.dropna(
    subset=feature_columns + ["target_sales"]
).copy()


print("Total forecasting records:", len(forecast_df))


# --------------------------------------------------
# 4. Chronological train-validation split
# --------------------------------------------------

train = forecast_df[
    forecast_df["target_date"] < "2017-01-01"
].copy()

valid = forecast_df[
    (forecast_df["target_date"] >= "2017-01-01") &
    (forecast_df["target_date"] <= "2017-12-31")
].copy()

X_train = train[feature_columns]
y_train = train["target_sales"]

X_valid = valid[feature_columns]
y_valid = valid["target_sales"]

print("Training records:", len(train))
print("Validation records:", len(valid))


# --------------------------------------------------
# 5. Train multi-horizon XGBoost
# --------------------------------------------------

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    random_state=42
)

print("\nTraining multi-horizon XGBoost model...")

model.fit(X_train, y_train)


# --------------------------------------------------
# 6. Generate validation predictions
# --------------------------------------------------

predictions = model.predict(X_valid)

# Sales cannot be negative
predictions = np.maximum(predictions, 0)

mae = mean_absolute_error(
    y_valid,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(y_valid, predictions)
)

wape = (
    np.abs(y_valid.to_numpy() - predictions).sum()
    / y_valid.sum()
) * 100


# --------------------------------------------------
# 7. Evaluate total 7-day demand
# --------------------------------------------------

valid["predicted_sales"] = predictions

weekly_results = (
    valid.groupby(
        ["date", "store", "item"],
        as_index=False
    )
    .agg(
        actual_7day_demand=("target_sales", "sum"),
        predicted_7day_demand=("predicted_sales", "sum")
    )
)

weekly_mae = mean_absolute_error(
    weekly_results["actual_7day_demand"],
    weekly_results["predicted_7day_demand"]
)

weekly_wape = (
    np.abs(
        weekly_results["actual_7day_demand"] -
        weekly_results["predicted_7day_demand"]
    ).sum()
    / weekly_results["actual_7day_demand"].sum()
) * 100


print("\n========== MULTI-HORIZON RESULTS ==========")
print(f"Daily MAE:       {mae:.4f}")
print(f"Daily RMSE:      {rmse:.4f}")
print(f"Daily WAPE:      {wape:.2f}%")

print("\n========== 7-DAY TOTAL DEMAND RESULTS ==========")
print(f"7-day MAE:       {weekly_mae:.4f}")
print(f"7-day WAPE:      {weekly_wape:.2f}%")


# --------------------------------------------------
# 8. Save trained model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/xgboost_multi_horizon.pkl"
)

print("\nMulti-horizon model saved successfully!")


# --------------------------------------------------
# 9. Forecast next 7 days from final dataset date
# --------------------------------------------------

latest_date = df["date"].max()

latest_rows = df[
    df["date"] == latest_date
].copy()

future_rows = []

for horizon in range(1, 8):

    temp = latest_rows.copy()

    temp["forecast_origin"] = latest_date

    temp["target_date"] = (
        latest_date +
        pd.Timedelta(days=horizon)
    )

    temp["target_year"] = temp["target_date"].dt.year
    temp["target_month"] = temp["target_date"].dt.month

    temp["target_day_of_week"] = (
        temp["target_date"].dt.day_of_week
    )

    temp["target_day_of_year"] = (
        temp["target_date"].dt.day_of_year
    )

    temp["target_weekend"] = (
        temp["target_day_of_week"] >= 5
    ).astype(int)

    temp["horizon"] = horizon

    future_rows.append(temp)


future_df = pd.concat(
    future_rows,
    ignore_index=True
)

future_predictions = model.predict(
    future_df[feature_columns]
)

future_df["predicted_sales"] = np.maximum(
    future_predictions, 0
)

# Save daily forecasts
forecast_output = future_df[
    [
        "store",
        "item",
        "forecast_origin",
        "target_date",
        "horizon",
        "predicted_sales"
    ]
].copy()

forecast_output.to_csv(
    "data/processed/future_7day_forecasts.csv",
    index=False
)

# Calculate total forecast demand for each store-item
demand_summary = (
    forecast_output.groupby(
        ["store", "item"],
        as_index=False
    )["predicted_sales"]
    .sum()
    .rename(
        columns={
            "predicted_sales": "forecast_7day_demand"
        }
    )
)

demand_summary.to_csv(
    "data/processed/future_7day_demand_summary.csv",
    index=False
)

print("\nFuture forecast origin:", latest_date)
print("Total daily forecast records:", len(forecast_output))
print("Total store-item combinations:", len(demand_summary))

print("\nSample 7-day demand forecasts:")
print(demand_summary.head())

print("\nForecast files saved successfully!")
