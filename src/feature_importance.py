
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# 1. Load trained XGBoost model
model = joblib.load("models/xgboost_model.pkl")

# 2. Define feature names
feature_columns = [
    "store", "item",
    "year", "month", "day", "day_of_week",
    "day_of_year", "weekend",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_30"
]

# 3. Extract feature importance
importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance", ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance_df.to_string(index=False))

# 4. Save feature importance
os.makedirs("data/processed", exist_ok=True)

importance_df.to_csv(
    "data/processed/feature_importance.csv",
    index=False
)

# 5. Visualize feature importance
plt.figure(figsize=(10, 7))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.gca().invert_yaxis()
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")
plt.tight_layout()

plt.savefig(
    "data/processed/feature_importance.png",
    dpi=300
)

plt.show()

print("\nFeature importance analysis completed!")
