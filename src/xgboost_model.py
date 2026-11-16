
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import joblib

# 1. Load feature-engineered dataset
df = pd.read_csv("data/processed/sales_features.csv")

df["date"] = pd.to_datetime(df["date"])

# 2. Remove rows with missing lag/rolling features
feature_columns = [
    "store", "item",
    "year", "month", "day", "day_of_week",
    "day_of_year", "weekend",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30"
]

df = df.dropna(subset=feature_columns + ["sales"]).copy()

# 3. Chronological train-validation split
train = df[df["date"] < "2017-01-01"].copy()
valid = df[df["date"] >= "2017-01-01"].copy()

X_train = train[feature_columns]
y_train = train["sales"]

X_valid = valid[feature_columns]
y_valid = valid["sales"]

print("Training records:", len(train))
print("Validation records:", len(valid))

# 4. Train XGBoost model
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

print("\nTraining XGBoost model...")
model.fit(X_train, y_train)

# 5. Generate predictions
predictions = model.predict(X_valid)

# 6. Evaluate model
mae = mean_absolute_error(y_valid, predictions)
rmse = np.sqrt(mean_squared_error(y_valid, predictions))
wape = np.sum(np.abs(y_valid - predictions)) / np.sum(y_valid) * 100

print("\n========== XGBOOST RESULTS ==========")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"WAPE: {wape:.2f}%")

# 7. Save trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgboost_model.pkl")

print("\nModel saved successfully!")
