
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "data" / "raw" / "train.csv")

df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# Average monthly sales for each year
monthly_yearly = df.pivot_table(
    index="year",
    columns="month",
    values="sales",
    aggfunc="mean"
)

print("\n========== MONTHLY SALES BY YEAR ==========")
print(monthly_yearly.round(2))

# Heatmap
plt.figure(figsize=(13, 6))

sns.heatmap(
    monthly_yearly,
    annot=True,
    fmt=".1f",
    cmap="YlGnBu"
)

plt.title("Monthly Sales Seasonality Across Years")
plt.xlabel("Month")
plt.ylabel("Year")
plt.tight_layout()
plt.show()

# Monthly trend comparison
plt.figure(figsize=(12, 6))

sns.lineplot(
    data=df.groupby(["year", "month"])["sales"].mean().reset_index(),
    x="month",
    y="sales",
    hue="year",
    marker="o"
)

plt.title("Monthly Sales Patterns Across Different Years")
plt.xlabel("Month")
plt.ylabel("Average Units Sold")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()

