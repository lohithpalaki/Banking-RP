import streamlit as st
import pandas as pd
import pickle

# Load the trained SVM model
with open('svc_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the required features
feature_cols = [
    'Age', 'Account_Balance', 'Transaction_Amount',
    'Account_Balance_After_Transaction', 'Loan_Amount', 'Interest_Rate',
    'Loan_Term', 'Credit_Limit', 'Credit_Card_Balance',
    'Minimum_Payment_Due', 'Rewards_Points'
]

st.title("Anomaly Prediction App")
st.markdown("Upload a CSV file with customer details to predict anomalies (-1 = Anomaly, 1 = Normal).")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if not all(col in data.columns for col in feature_cols):
            st.error("Uploaded file must contain all required columns.")
            st.write("Required columns:", feature_cols)
        else:
            X_input = data[feature_cols]
            prediction = model.predict(X_input)
            data['Prediction'] = prediction
            st.success("Prediction completed!")
            st.dataframe(data)

            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv'
            )
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("📥 [Download Sample Input Template](sample_input_template.csv)")