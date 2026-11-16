
import pandas as pd
import numpy as np
import os
import joblib

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Load feature-engineered dataset
df = pd.read_csv("data/processed/sales_features.csv")
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["store", "item", "date"]).copy()

# 2. Create a 7-day-ahead target for each store-item series
df["target_sales"] = (
    df.groupby(["store", "item"])["sales"].shift(-7)
)

df["target_date"] = df["date"] + pd.Timedelta(days=7)

# 3. Create calendar features for the forecast target date
df["target_year"] = df["target_date"].dt.year
df["target_month"] = df["target_date"].dt.month
df["target_day_of_week"] = df["target_date"].dt.day_of_week
df["target_day_of_year"] = df["target_date"].dt.day_of_year

df["target_weekend"] = (
    df["target_day_of_week"] >= 5
).astype(int)

# 4. Define features available at the forecast origin
feature_columns = [
    "store", "item",
    "target_year", "target_month",
    "target_day_of_week", "target_day_of_year",
    "target_weekend",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14",
    "rolling_mean_30"
]

# 5. Remove rows with missing values
df = df.dropna(
    subset=feature_columns + ["target_sales"]
).copy()

# 6. Chronological split based on target date
train = df[df["target_date"] < "2017-01-01"].copy()

valid = df[
    (df["target_date"] >= "2017-01-01") &
    (df["target_date"] <= "2017-12-31")
].copy()

X_train = train[feature_columns]
y_train = train["target_sales"]

X_valid = valid[feature_columns]
y_valid = valid["target_sales"]

print("Training records:", len(train))
print("Validation records:", len(valid))

# 7. Train XGBoost
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

print("\nTraining 7-day XGBoost model...")

model.fit(X_train, y_train)

# 8. Generate predictions
predictions = model.predict(X_valid)

# 9. Evaluate model
mae = mean_absolute_error(y_valid, predictions)

rmse = np.sqrt(
    mean_squared_error(y_valid, predictions)
)

wape = (
    np.abs(y_valid - predictions).sum()
    / y_valid.sum()
) * 100

print("\n========== 7-DAY FORECAST RESULTS ==========")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"WAPE: {wape:.2f}%")

# 10. Save model
os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/xgboost_7day_model.pkl"
)

print("\n7-day XGBoost model saved successfully!")
