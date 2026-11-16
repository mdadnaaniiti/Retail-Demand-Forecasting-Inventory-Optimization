
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

# 1. Load dataset and trained model
df = pd.read_csv("data/processed/sales_features.csv")
df["date"] = pd.to_datetime(df["date"])

model = joblib.load("models/xgboost_model.pkl")

# 2. Define model features
feature_columns = [
    "store", "item",
    "year", "month", "day", "day_of_week",
    "day_of_year", "weekend",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30"
]

# 3. Prepare validation dataset
df = df.dropna(subset=feature_columns + ["sales"]).copy()

valid = df[df["date"] >= "2017-01-01"].copy()

X_valid = valid[feature_columns]

# 4. Generate predictions
valid["predicted_sales"] = model.predict(X_valid)

# 5. Calculate residuals
# Positive residual = underforecasting
# Negative residual = overforecasting
valid["residual"] = (
    valid["sales"] - valid["predicted_sales"]
)

valid["absolute_error"] = valid["residual"].abs()

# 6. Analyze prediction errors
mean_residual = valid["residual"].mean()
mae = valid["absolute_error"].mean()

underforecast = (valid["residual"] > 0).sum()
overforecast = (valid["residual"] < 0).sum()
exact_predictions = (valid["residual"] == 0).sum()

print("\n========== RESIDUAL ANALYSIS ==========")

print(f"Mean Residual: {mean_residual:.4f}")
print(f"MAE: {mae:.4f}")

print(f"\nUnderforecasting observations: {underforecast}")
print(f"Overforecasting observations: {overforecast}")
print(f"Exact predictions: {exact_predictions}")

print(
    f"\nUnderforecasting percentage: "
    f"{underforecast / len(valid) * 100:.2f}%"
)

print(
    f"Overforecasting percentage: "
    f"{overforecast / len(valid) * 100:.2f}%"
)

# 7. Save residual data
os.makedirs("data/processed", exist_ok=True)

valid.to_csv(
    "data/processed/residual_analysis.csv",
    index=False
)

# 8. Residual distribution
plt.figure(figsize=(10, 5))

plt.hist(valid["residual"], bins=50, edgecolor="black")

plt.axvline(
    x=0,
    linestyle="--",
    label="Zero Error"
)

plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Distribution of XGBoost Forecasting Residuals")
plt.legend()
plt.tight_layout()

plt.savefig(
    "data/processed/residual_distribution.png",
    dpi=300
)

plt.show()

# 9. Residuals versus predicted sales
plt.figure(figsize=(10, 5))

plt.scatter(
    valid["predicted_sales"],
    valid["residual"],
    alpha=0.15
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Sales")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted Sales")
plt.tight_layout()

plt.savefig(
    "data/processed/residual_vs_predicted.png",
    dpi=300
)

plt.show()

print("\nResidual analysis completed successfully!")
