import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('C:/Users/91981/customer_churn_prediction_system/data/churn_data.csv.csv')
print("Initial rows:", len(df))

# Replace blanks with NaN
df.replace(" ", np.nan, inplace=True)
print("After replacing blanks with NaN:", len(df))

# Drop rows where 'Churn' is missing
print("Churn column value counts:\n", df["Churn"].value_counts(dropna=False))
df = df[df['Churn'].notna()]
print("After dropping NaNs:", len(df))

# Drop customerID if it exists
if 'customerID' in df.columns:
    df.drop(columns=['customerID'], inplace=True)

# Convert 'TotalCharges' to numeric (force non-convertible to NaN then drop)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

# Encode target
df['Churn'] = df['Churn'].map({'Yes': True, 'No': False})

# One-hot encode categorical columns
df = pd.get_dummies(df, drop_first=True)
print("After one-hot encoding:", len(df))

# Split features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE on training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Before SMOTE:\n", y_train.value_counts())
print("After SMOTE:\n", y_train_resampled.value_counts())

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train_resampled)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
os.makedirs("../artifacts", exist_ok=True)
joblib.dump(model, "../artifacts/rf_model.pkl")
joblib.dump(scaler, "../artifacts/scaler.pkl")
print("\n✅ Model and scaler saved.")

# Save feature column names
import pickle

columns = X.columns.tolist()
os.makedirs("../models", exist_ok=True)
with open("../models/columns.pkl", "wb") as f:
    pickle.dump(columns, f)

print("\n✅ Column names saved to ../models/columns.pkl")
