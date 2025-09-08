import streamlit as st
import pickle
import pandas as pd
import numpy as np

with open("best_model.pkl", "rb") as file:
    best_model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

with open("imputer.pkl", "rb") as file:
    imputer = pickle.load(file)

if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":
    st.title("Credit Card Fraud Detector")
    st.write("""
             This app predicts whether a credit card transaction is fraudulent or legitimate.
             It uses a machine learning model trained on historical transaction data to help prevent financial fraud.
             Enter the transaction details below to get an instant fraud prediction    
                     """)
    
    all_columns = ["Time", "V1","V2","V3","V4","V5","V6","V7","V8",
               "V9","V10","V11","V12","V13","V14","V15","V16","V17","V18",
               "V19","V20","V21","V22","V23","V24","V25","V26","V27","V28",
               "Amount"]

    input_dict = {}

    for col in all_columns:
       value = st.number_input(col, value= 0.00, format="%.6f")

       if value == 0.0:
        input_dict[col] = np.nan
       else:
        input_dict[col] = value


    if st.button("Predict"):

        user_df = pd.DataFrame([input_dict])

        user_df = imputer.transform(user_df)
        user_df = scaler.transform(user_df)
        prediction = best_model.predict(user_df)

        if prediction[0] == 1:
            st.error("⚠️ This Transcation is likely FRAUDULENT")
        else:
            st.success("✅ This transaction appears LEGITIMATE.")


 









                           
    
    