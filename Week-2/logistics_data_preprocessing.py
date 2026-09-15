# NSDC Logistics Data Analyst Internship - Week 2
# Data Collection, Cleaning and Preprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dataset
df = pd.read_csv("Delivery_Logistics.csv")

print("Shape:", df.shape)
print(df.head())
print(df.info())

# 2. Standardize column names
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
      .str.replace(r"[^a-z0-9_]", "", regex=True)
)

# 3. Check data quality
print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False))

print("\nDuplicate rows:", df.duplicated().sum())

# 4. Remove duplicate rows
df = df.drop_duplicates().copy()

# 5. Convert numeric columns
numeric_candidates = [
    "distance", "package_weight", "actual_delivery_time",
    "expected_delivery_time", "delivery_time",
    "expected_time", "delivery_cost", "rating"
]

for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 6. Handle missing values
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include="object").columns

for col in num_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    if df[col].isna().any():
        df[col] = df[col].fillna("Unknown")

# 7. Remove invalid negative values
non_negative_cols = [
    "distance", "package_weight", "actual_delivery_time",
    "expected_delivery_time", "delivery_time",
    "expected_time", "delivery_cost"
]

for col in non_negative_cols:
    if col in df.columns:
        df = df[df[col] >= 0]

# 8. Outlier detection using IQR
def iqr_bounds(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

for col in [
    "distance", "package_weight", "delivery_cost",
    "actual_delivery_time", "expected_delivery_time"
]:
    if col in df.columns:
        lower, upper = iqr_bounds(df[col])
        mask = (df[col] < lower) | (df[col] > upper)
        print(f"{col}: {mask.sum()} potential outliers")

# 9. Create delay features
if {"actual_delivery_time", "expected_delivery_time"}.issubset(df.columns):
    df["delay_duration"] = (
        df["actual_delivery_time"] - df["expected_delivery_time"]
    )
    df["delay_flag"] = (df["delay_duration"] > 0).astype(int)

# 10. Standardization
scale_cols = [
    c for c in [
        "distance", "package_weight", "delivery_cost",
        "actual_delivery_time", "expected_delivery_time"
    ] if c in df.columns
]

if scale_cols:
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])

# 11. Train/test split example
if "delay_flag" in df.columns:
    X = df.drop(columns=["delay_flag"])
    y = df["delay_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

# 12. Save cleaned dataset
df.to_csv("delivery_logistics_cleaned.csv", index=False)

print("\nSaved: delivery_logistics_cleaned.csv")
print("Preprocessing completed.")
