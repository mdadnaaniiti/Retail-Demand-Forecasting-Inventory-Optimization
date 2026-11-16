
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

results = [
    {
        "model": "Naive",
        "MAE": 11.2447,
        "RMSE": 15.3892,
        "WAPE": 19.12
    },
    {
        "model": "Seasonal Naive",
        "MAE": 8.8415,
        "RMSE": 11.5967,
        "WAPE": 15.03
    }
]

comparison = pd.DataFrame(results)

output_path = BASE_DIR / "data" / "processed"
output_path.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    output_path / "model_comparison.csv",
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

print("\nBest model by WAPE:")
print(
    comparison.loc[
        comparison["WAPE"].idxmin()
    ].to_string()
)

print("\nComparison saved successfully!")
