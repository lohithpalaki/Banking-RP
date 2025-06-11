import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Dummy login credentials
auth_users = {"Risk_Admin": "Secure123"}

# Session management
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

# Predict one sample
def single_prediction_page():
    st.title("📌 Predict Single Customer Risk")

    total_spent = st.number_input("Total Spent", 0.0, 1e7, 5000.0)
    loyalty_points = st.number_input("Loyalty Points Earned", 0, 100000, 250)
    referrals = st.number_input("Referral Count", 0, 1000, 5)
    cashback = st.number_input("Cashback Received", 0.0, 100000.0, 300.0)
    satisfaction = st.slider("Customer Satisfaction Score", 0.0, 10.0, 7.5)

    if st.button("Predict Risk"):
        model = joblib.load("lgbm_model.pkl")
        input_data = np.array([[total_spent, loyalty_points, referrals, cashback, satisfaction]])
        prediction = model.predict(input_data)[0]
        label_map = {0: "High", 1: "Low", 2: "Medium"}
        st.success(f"Predicted Risk Category: {label_map[prediction]}")

# Predict batch from CSV
def batch_prediction_page():
    st.title("📂 Predict Risk for Batch (CSV Upload)")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data", df.head())

        expected_cols = ['Total_Spent', 'Loyalty_Points_Earned', 'Referral_Count', 'Cashback_Received', 'Customer_Satisfaction_Score']
        if all(col in df.columns for col in expected_cols):
            model = joblib.load("lgbm_model.pkl")
            preds = model.predict(df[expected_cols])
            label_map = {0: "High", 1: "Low", 2: "Medium"}
            df['Predicted_Risk'] = [label_map[p] for p in preds]
            st.write("📊 Predictions", df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇ Download Predictions", csv, "predicted_risks.csv", "text/csv")
        else:
            st.error(f"CSV must contain columns: {', '.join(expected_cols)}")

# Router
if st.session_state.page == "Login":
    login_page()
elif st.session_state.page == "Predict One":
    single_prediction_page()
elif st.session_state.page == "Predict from CSV":
    batch_prediction_page()
