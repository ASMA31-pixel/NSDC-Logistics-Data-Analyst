# NSDC Logistics Data Analyst Internship - Week 3
# Advanced Data Analysis and Visualization

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("Delivery_Logistics.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# 2. Basic information
print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# 3. Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^a-z0-9_]", "", regex=True)
)

print("\nColumns:")
print(df.columns.tolist())

# 4. Distribution of numerical variables
numeric_cols = df.select_dtypes(include="number").columns

for col in numeric_cols:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# 5. Correlation analysis
if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df[numeric_cols].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Correlation Matrix of Numerical Variables")
    plt.tight_layout()
    plt.show()

# 6. Distance vs delivery time
if {"distance", "actual_delivery_time"}.issubset(df.columns):
    plt.figure(figsize=(7, 4))
    sns.scatterplot(
        data=df,
        x="distance",
        y="actual_delivery_time"
    )
    plt.title("Distance vs Actual Delivery Time")
    plt.xlabel("Distance")
    plt.ylabel("Actual Delivery Time")
    plt.tight_layout()
    plt.show()

# 7. Delivery cost distribution
if "delivery_cost" in df.columns:
    plt.figure(figsize=(7, 4))
    sns.boxplot(y=df["delivery_cost"])
    plt.title("Delivery Cost Distribution")
    plt.ylabel("Delivery Cost")
    plt.tight_layout()
    plt.show()

# 8. Create delay features
if {"actual_delivery_time", "expected_delivery_time"}.issubset(df.columns):
    df["delay_duration"] = (
        df["actual_delivery_time"] -
        df["expected_delivery_time"]
    )

    df["delay_flag"] = (df["delay_duration"] > 0).astype(int)

    print("\nDelay summary:")
    print(df["delay_flag"].value_counts())

    # Delay visualization
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="delay_flag")
    plt.title("Delayed vs Non-Delayed Deliveries")
    plt.xlabel("Delay Flag")
    plt.ylabel("Number of Deliveries")
    plt.tight_layout()
    plt.show()

# 9. Save analysis-ready data
df.to_csv("delivery_logistics_analysis_ready.csv", index=False)

print("\nAnalysis completed.")
print("Saved: delivery_logistics_analysis_ready.csv")
