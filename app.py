import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('svc_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the input fields (in same order as training)
feature_cols = [
    'Age', 'Account_Balance', 'Transaction_Amount',
    'Account_Balance_After_Transaction', 'Loan_Amount', 'Interest_Rate',
    'Loan_Term', 'Credit_Limit', 'Credit_Card_Balance',
    'Minimum_Payment_Due', 'Rewards_Points'
]

st.title("Customer Anomaly Prediction")

# Option to upload CSV or manual entry
option = st.radio("Choose input method:", ["Manual Entry", "Upload CSV"])

if option == "Manual Entry":
    user_input = {}
    for col in feature_cols:
        user_input[col] = st.number_input(f"{col}", value=0.0)
    input_df = pd.DataFrame([user_input])
else:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)

# Predict when data is ready
if 'input_df' in locals():
    input_scaled = scaler.transform(input_df)
    predictions = model.predict(input_scaled)
    input_df['Predicted_Anomaly'] = predictions
    st.subheader("Prediction Results")
    st.write(input_df)
    st.download_button("Download Results", input_df.to_csv(index=False), "predictions.csv", "text/csv")
