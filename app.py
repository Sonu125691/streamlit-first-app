import streamlit as st
import pickle
import pandas as pd
import numpy as np

with open("final_scaler.pkl", "rb") as file:
   scaler = pickle.load(file)

with open("final_model.pkl", "rb") as file:
   model = pickle.load(file)

if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.page == "Home":
    st.title("Credit Card Fraud Detector")
    st.write("""
             This app predicts whether a credit card transaction is fraudulent or legitimate.
             It uses a machine learning model trained on historical transaction data to help prevent financial fraud.
             Enter the transaction details below to get an instant fraud prediction    
                     """)
    
    V_columns = ["V1","V2","V3","V4","V5","V6","V7","V8",
               "V9","V10","V11","V12","V13","V14","V15","V16","V17","V18",
               "V19","V20","V21","V22","V23","V24","V25","V26","V27","V28"]

    input_dict = {}

    for col in V_columns:
       value = st.number_input(col, value= 1.00, format="%.6f")
       input_dict[col] = value

    amount = st.number_input("Amount", 0, 100000, 100)
    input_dict["Amount"] = amount
                    
    if st.button("Predict"):

        user_df = pd.DataFrame([input_dict])
        user_df["Amount"] = scaler.transform(user_df[["Amount"]])
        prediction = model.predict(user_df)

        if prediction[0] == 1:
            st.error("⚠️ This Transcation is likely FRAUDULENT")
        else:
            st.success("✅ This transaction appears LEGITIMATE.")


 









                           
    
    