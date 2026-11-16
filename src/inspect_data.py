
import pandas as pd
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
TRAIN_PATH = BASE_DIR / "data" / "raw" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "raw" / "test.csv"
SUBMISSION_PATH = BASE_DIR / "data" / "raw" / "sample_submission.csv"

# Load datasets
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
submission = pd.read_csv(SUBMISSION_PATH)

# Basic information
print("\n========== DATASET SHAPES ==========")
print("Train:", train.shape)
print("Test:", test.shape)
print("Submission:", submission.shape)

print("\n========== TRAIN DATA ==========")
print(train.head())

print("\n========== DATA TYPES ==========")
print(train.dtypes)

print("\n========== MISSING VALUES ==========")
print(train.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(train.duplicated().sum())

print("\n========== DATE INFORMATION ==========")
train["date"] = pd.to_datetime(train["date"])

print("Start date:", train["date"].min())
print("End date:", train["date"].max())

print("\n========== UNIQUE STORES AND ITEMS ==========")
print("Number of stores:", train["store"].nunique())
print("Number of items:", train["item"].nunique())

print("\n========== SALES STATISTICS ==========")
print(train["sales"].describe())
