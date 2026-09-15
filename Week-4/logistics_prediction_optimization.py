# NSDC Logistics Data Analyst Internship - Week 4
# Predictive Modeling and Optimization

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("Delivery_Logistics.csv")

# 2. Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace(r"[^a-z0-9_]", "", regex=True)
)

# 3. Create target variable
if {"actual_delivery_time", "expected_delivery_time"}.issubset(df.columns):
    df["delay_flag"] = (
        df["actual_delivery_time"] >
        df["expected_delivery_time"]
    ).astype(int)

# 4. Select numerical features
feature_candidates = [
    "distance",
    "package_weight",
    "actual_delivery_time",
    "expected_delivery_time",
    "delivery_cost",
    "rating"
]

features = [col for col in feature_candidates if col in df.columns]

if "delay_flag" in df.columns and len(features) > 0:

    X = df[features].copy()
    y = df["delay_flag"]

    # Handle missing values
    X = X.fillna(X.median())

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # 6. Build Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 7. Make predictions
    y_pred = model.predict(X_test)

    # 8. Evaluate model
    accuracy = accuracy_score(y_test, y_pred)

    print("Model Accuracy:", round(accuracy, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 9. Feature importance
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print("\nFeature Importance:")
    print(importance)

    # 10. Simple optimization recommendations
    print("\nOptimization Recommendations:")
    print("- Monitor routes with higher delivery distances.")
    print("- Prioritize deliveries predicted to be delayed.")
    print("- Review delivery costs for inefficient routes.")
    print("- Use model insights to support delivery planning.")

else:
    print("Required columns for predictive modeling were not found.")

print("\nPredictive modeling and optimization analysis completed.")
