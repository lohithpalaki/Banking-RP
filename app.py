import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Dummy login credentials
auth_users = {"Risk_Admin": "Secure123"}

# Session state management
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Login"

# Sidebar navigation
if st.session_state.logged_in:
    st.sidebar.title("Navigation")
    st.session_state.page = st.sidebar.selectbox(
        "Go to", 
        ["Predict One", "Predict from CSV"], 
        index=["Predict One", "Predict from CSV"].index(st.session_state.page)
        if st.session_state.page in ["Predict One", "Predict from CSV"] else 0
    )
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "Login"
        st.experimental_rerun()
else:
    st.session_state.page = "Login"

# Login Page
def login_page():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in auth_users and auth_users[username] == password:
            st.session_state.logged_in = True
            st.session_state.page = "Predict One"
            st.success("Login successful!")
        else:
            st.error("Invalid credentials")

# Predict single input
def single_prediction_page():
    st.title("📌 Predict Customer Risk (Single Entry)")

    Age = st.number_input("Age", 18, 100, 30)
    Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    Contact_Number = st.text_input("Contact Number", "9876543210")
    Account_Type = st.selectbox("Account Type", ["Savings", "Current"])
    Account_Balance = st.number_input("Account Balance", 0.0, 1e7, 50000.0)
    Transaction_Type = st.selectbox("Transaction Type", ["Credit", "Debit"])
    Transaction_Amount = st.number_input("Transaction Amount", 0.0, 1e6, 10000.0)
    Account_Balance_After_Transaction = st.number_input("Balance After Transaction", 0.0, 1e7, 40000.0)
    Anomaly = st.selectbox("Anomaly Detected?", ["1", "-1"])
    Loan_Amount = st.number_input("Loan Amount", 0.0, 1e7, 200000.0)
    Loan_Type = st.selectbox("Loan Type", ["Auto", "Personal", "Mortgage"])
    Interest_Rate = st.slider("Interest Rate (%)", 0.0, 25.0, 8.5)
    Loan_Term = st.number_input("Loan Term (months)", 1, 360, 60)
    Loan_Status = st.selectbox("Loan Status", ["Approved", "Rejected", "Pending"])
    Card_Type = st.selectbox("Card Type", ["MasterCard", "Visa", "AMEX"])
    Credit_Limit = st.number_input("Credit Limit", 0.0, 1e6, 100000.0)
    Credit_Card_Balance = st.number_input("Credit Card Balance", 0.0, 1e6, 20000.0)
    Minimum_Payment_Due = st.number_input("Minimum Payment Due", 0.0, 1e6, 5000.0)
    Rewards_Points = st.number_input("Reward Points", 0, 100000, 1200)
    Feedback_Type = st.selectbox("Feedback Type", ["Complaint", "Suggestion", "Praise"])
    Resolution_Status = st.selectbox("Resolution Status", ["Resolved", "Pending"])

    if st.button("Predict Risk"):
        model = joblib.load("lgbm_model.pkl")

        input_data = pd.DataFrame([{
            'Age': Age,
            'Gender': Gender,
            'Contact_Number': Contact_Number,
            'Account_Type': Account_Type,
            'Account_Balance': Account_Balance,
            'Transaction_Type': Transaction_Type,
            'Transaction_Amount': Transaction_Amount,
            'Account_Balance_After_Transaction': Account_Balance_After_Transaction,
            'Anomaly': Anomaly,
            'Loan_Amount': Loan_Amount,
            'Loan_Type': Loan_Type,
            'Interest_Rate': Interest_Rate,
            'Loan_Term': Loan_Term,
            'Loan_Status': Loan_Status,
            'Card_Type': Card_Type,
            'Credit_Limit': Credit_Limit,
            'Credit_Card_Balance': Credit_Card_Balance,
            'Minimum_Payment_Due': Minimum_Payment_Due,
            'Rewards_Points': Rewards_Points,
            'Feedback_Type': Feedback_Type,
            'Resolution_Status': Resolution_Status
        }])

        # Encode categorical columns
        for col in input_data.select_dtypes(include='object').columns:
            input_data[col] = input_data[col].astype("category").cat.codes

        prediction = model.predict(input_data)[0]
        label_map = {0: "High", 1: "Low", 2: "Medium"}
        st.success(f"Predicted Risk Category: {label_map[prediction]}")

# Predict from uploaded CSV
def batch_prediction_page():
    st.title("📂 Predict Risk from CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data", df.head())

        expected_cols = ['Age', 'Gender', 'Contact_Number', 'Account_Type', 'Account_Balance',
                         'Transaction_Type', 'Transaction_Amount', 'Account_Balance_After_Transaction', 'Anomaly',
                         'Loan_Amount', 'Loan_Type', 'Interest_Rate', 'Loan_Term', 'Loan_Status',
                         'Card_Type', 'Credit_Limit', 'Credit_Card_Balance', 'Minimum_Payment_Due',
                         'Rewards_Points', 'Feedback_Type', 'Resolution_Status']

        if all(col in df.columns for col in expected_cols):
            model = joblib.load("lgbm_model.pkl")
            df_encoded = df.copy()
            for col in df_encoded.select_dtypes(include='object').columns:
                df_encoded[col] = df_encoded[col].astype("category").cat.codes
            preds = model.predict(df_encoded[expected_cols])
            label_map = {0: "High", 1: "Low", 2: "Medium"}
            df['Predicted_Risk'] = [label_map[p] for p in preds]
            st.write("📊 Predictions", df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download Predictions", csv, "predicted_risks.csv", "text/csv")
        else:
            st.error(f"CSV must contain required columns. Found: {list(df.columns)}")

# Route the page
if st.session_state.page == "Login":
    login_page()
elif st.session_state.page == "Predict One":
    single_prediction_page()
elif st.session_state.page == "Predict from CSV":
    batch_prediction_page()
