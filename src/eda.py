
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load data
df = pd.read_csv(BASE_DIR / "data" / "raw" / "train.csv")

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Overall daily sales
daily_sales = df.groupby("date")["sales"].sum()

# 1. Overall sales trend
plt.figure(figsize=(14, 5))
plt.plot(daily_sales.index, daily_sales.values)
plt.title("Overall Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Units Sold")
plt.tight_layout()
plt.show()

# 2. Average sales by store
store_sales = df.groupby("store")["sales"].mean()

plt.figure(figsize=(10, 5))
sns.barplot(x=store_sales.index, y=store_sales.values)
plt.title("Average Sales by Store")
plt.xlabel("Store")
plt.ylabel("Average Units Sold")
plt.tight_layout()
plt.show()

# 3. Average sales by product
item_sales = df.groupby("item")["sales"].mean()

plt.figure(figsize=(14, 5))
sns.barplot(x=item_sales.index, y=item_sales.values)
plt.title("Average Sales by Product")
plt.xlabel("Product")
plt.ylabel("Average Units Sold")
plt.tight_layout()
plt.show()

# 4. Monthly sales pattern
df["month"] = df["date"].dt.month
monthly_sales = df.groupby("month")["sales"].mean()

plt.figure(figsize=(10, 5))
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker="o")
plt.title("Average Sales by Month")
plt.xlabel("Month")
plt.ylabel("Average Units Sold")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()

print("\nEDA completed successfully!")
