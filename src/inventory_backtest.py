
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Load data and trained model
# --------------------------------------------------

df = pd.read_csv("data/processed/sales_features.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["store", "item", "date"]
).copy()

model = joblib.load(
    "models/xgboost_multi_horizon.pkl"
)


# --------------------------------------------------
# 2. Define model features
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

feature_columns = base_features + [
    "target_year",
    "target_month",
    "target_day_of_week",
    "target_day_of_year",
    "target_weekend",
    "horizon"
]


# --------------------------------------------------
# 3. Generate rolling validation forecasts
# --------------------------------------------------

backtest_results = []

for horizon in range(1, 8):

    temp = df.copy()

    temp["actual_sales"] = (
        temp.groupby(["store", "item"])["sales"]
        .shift(-horizon)
    )

    temp["target_date"] = (
        temp["date"] +
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

    # Use the same weekly forecast-origin convention
    temp = temp[
        temp["date"].map(
            lambda x: x.toordinal() % 7 == 0
        )
    ]

    # Keep only 2017 target dates
    temp = temp[
        (temp["target_date"] >= "2017-01-01") &
        (temp["target_date"] <= "2017-12-31")
    ].copy()

    temp = temp.dropna(
        subset=feature_columns + ["actual_sales"]
    )

    predictions = model.predict(
        temp[feature_columns]
    )

    temp["predicted_sales"] = np.maximum(
        predictions, 0
    )

    backtest_results.append(
        temp[
            [
                "date",
                "target_date",
                "store",
                "item",
                "horizon",
                "actual_sales",
                "predicted_sales"
            ]
        ]
    )


# --------------------------------------------------
# 4. Combine all forecasts
# --------------------------------------------------

results = pd.concat(
    backtest_results,
    ignore_index=True
)

print("Total validation predictions:", len(results))


# --------------------------------------------------
# 5. Daily forecast evaluation
# --------------------------------------------------

mae = mean_absolute_error(
    results["actual_sales"],
    results["predicted_sales"]
)

rmse = np.sqrt(
    mean_squared_error(
        results["actual_sales"],
        results["predicted_sales"]
    )
)

wape = (
    np.abs(
        results["actual_sales"] -
        results["predicted_sales"]
    ).sum()
    / results["actual_sales"].sum()
) * 100


# --------------------------------------------------
# 6. Aggregate 7-day demand by forecast origin
# --------------------------------------------------

weekly = (
    results.groupby(
        ["date", "store", "item"],
        as_index=False
    )
    .agg(
        actual_7day_demand=("actual_sales", "sum"),
        predicted_7day_demand=("predicted_sales", "sum")
    )
)

weekly_mae = mean_absolute_error(
    weekly["actual_7day_demand"],
    weekly["predicted_7day_demand"]
)

weekly_wape = (
    np.abs(
        weekly["actual_7day_demand"] -
        weekly["predicted_7day_demand"]
    ).sum()
    / weekly["actual_7day_demand"].sum()
) * 100


# --------------------------------------------------
# 7. Evaluate each forecast horizon
# --------------------------------------------------

horizon_results = (
    results.groupby("horizon")
    .apply(
        lambda x: pd.Series({
            "MAE": mean_absolute_error(
                x["actual_sales"],
                x["predicted_sales"]
            ),
            "WAPE": (
                np.abs(
                    x["actual_sales"] -
                    x["predicted_sales"]
                ).sum()
                / x["actual_sales"].sum()
            ) * 100
        }),
        include_groups=False
    )
    .reset_index()
)


# --------------------------------------------------
# 8. Display results
# --------------------------------------------------

print("\n========== ROLLING BACKTEST RESULTS ==========")

print(f"Daily MAE:  {mae:.4f}")
print(f"Daily RMSE: {rmse:.4f}")
print(f"Daily WAPE: {wape:.2f}%")

print("\n========== 7-DAY TOTAL DEMAND ==========")

print(f"7-day MAE:  {weekly_mae:.4f}")
print(f"7-day WAPE: {weekly_wape:.2f}%")

print("\n========== HORIZON-WISE PERFORMANCE ==========")
print(horizon_results.to_string(index=False))


# --------------------------------------------------
# 9. Save backtesting results
# --------------------------------------------------

results.to_csv(
    "data/processed/inventory_backtest_daily.csv",
    index=False
)

weekly.to_csv(
    "data/processed/inventory_backtest_weekly.csv",
    index=False
)

horizon_results.to_csv(
    "data/processed/inventory_backtest_horizons.csv",
    index=False
)

print("\nBacktesting completed successfully!")
print("Results saved in data/processed/")
