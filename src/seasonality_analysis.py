
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

df = pd.read_csv(BASE_DIR / "data" / "raw" / "train.csv")

df["date"] = pd.to_datetime(df["date"])

# Extract calendar information
df["day_of_week"] = df["date"].dt.dayofweek
df["day_name"] = df["date"].dt.day_name()

df["year"] = df["date"].dt.year

# Weekly sales pattern
weekly_sales = df.groupby("day_of_week")["sales"].mean()

plt.figure(figsize=(10, 5))
sns.barplot(x=weekly_sales.index, y=weekly_sales.values)

plt.title("Average Sales by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Units Sold")

plt.xticks(
    range(7),
    ["Monday", "Tuesday", "Wednesday",
     "Thursday", "Friday", "Saturday", "Sunday"]
)

plt.tight_layout()
plt.show()

# Yearly sales comparison
yearly_sales = df.groupby("year")["sales"].mean()

plt.figure(figsize=(10, 5))
sns.lineplot(
    x=yearly_sales.index,
    y=yearly_sales.values,
    marker="o"
)

plt.title("Average Sales by Year")
plt.xlabel("Year")
plt.ylabel("Average Units Sold")

plt.tight_layout()
plt.show()

print("\nWeekly sales pattern:")
print(weekly_sales)

print("\nYearly average sales:")
print(yearly_sales)
