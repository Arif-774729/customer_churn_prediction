import sqlite3
import pandas as pd

csv_file_path = '../data/churn_data.csv'

conn = sqlite3.connect('../db/churn_data.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS churn_data (
    customerID TEXT PRIMARY KEY,
    gender TEXT,
    SeniorCitizen INTEGER,
    Partner TEXT,
    Dependents TEXT,
    tenure INTEGER,
    PhoneService TEXT,
    MultipleLines TEXT,
    InternetService TEXT,
    OnlineSecurity TEXT,
    OnlineBackup TEXT,
    DeviceProtection TEXT,
    TechSupport TEXT,
    StreamingTV TEXT,
    StreamingMovies TEXT,
    Contract TEXT,
    PaperlessBilling TEXT,
    PaymentMethod TEXT,
    MonthlyCharges REAL,
    TotalCharges REAL,
    Churn TEXT
)
''')


df = pd.read_csv('C:/Users/91981/customer_churn_prediction_system/data/churn_data.csv.csv')



df.to_sql('churn_data', conn, if_exists='replace', index=False)

conn.commit()
conn.close()

print("Database setup completed and data loaded successfully.")
