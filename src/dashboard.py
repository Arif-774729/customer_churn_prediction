import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

import joblib
import pickle


model = joblib.load('../artifacts/rf_model.pkl')


scaler = joblib.load('../artifacts/scaler.pkl')


with open('../models/columns.pkl', 'rb') as f:
    columns = pickle.load(f)


st.title("📊 Customer Churn Prediction Dashboard")

st.markdown("Use this dashboard to predict whether a customer is likely to churn based on input features.")


st.header("📥 Customer Details")
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner", ["No", "Yes"])
dependents = st.selectbox("Has Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=2500.0)

# Process categorical inputs
input_dict = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "Contract": contract,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# One-hot encode
categorical = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'Contract', 'PaymentMethod']
input_df = pd.get_dummies(input_df, columns=categorical)

# Load dummy columns from training for alignment
with open('../models/columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_columns]

# Scale
scaled_input = scaler.transform(input_df)

# Predict
if st.button("🚀 Predict Churn"):
    pred = model.predict(scaled_input)[0]
    prob = model.predict_proba(scaled_input)[0][1]
    st.subheader("🔍 Prediction Result:")
    st.write(f"**Churn Prediction:** {'Yes' if pred else 'No'}")
    st.write(f"**Churn Probability:** {prob:.2%}")

    # Conditional color
    if pred:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

