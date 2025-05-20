import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_csv('C:/Users/91981/customer_churn_prediction_system/data/churn_data.csv.csv')

# Fill missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical columns
le = LabelEncoder()
categorical_columns = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'Churn'
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Save preprocessed data
df.to_csv('../data/preprocessed_data.csv', index=False)
print("✅ Data preprocessing completed and saved to 'preprocessed_data.csv'")
