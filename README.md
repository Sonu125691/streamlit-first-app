# 💳 Credit Card Fraud Detector App

A machine learning-powered application that predicts whether a credit card transaction is **fraudulent** or **legitimate**.  
The dataset used in this project is a real-world, **highly imbalanced dataset** of credit card transactions, with 29 features including 28 anonymized PCA-transformed variables (`V1–V28`) and the transaction `Amount`. This imbalance makes F1 score a critical metric for evaluating model performance.

---

## 📝 Project Overview

In this project, I implemented **six different machine learning models** to ensure robust performance:  

- **Logistic Regression**  
- **Support Vector Classifier (SVC)**  
- **Decision Tree**  
- **Random Forest**  
- **XGBoost**  
- **Gaussian Naive Bayes**  

Each model was trained on the 29-feature dataset. The `Amount` feature was scaled using **StandardScaler** to align with the other PCA-transformed features. After evaluation, I selected **XGBoost Classifier** as the final model, achieving a **high F1 score of 0.8865**, making it the most reliable choice for this imbalanced dataset.  

The final model is deployed in a **Streamlit web application**, where users can input transaction details and receive **instant predictions** indicating whether a transaction is **Fraudulent** or **Legitimate**.

---

## 🚀 Features

- Robust machine learning model selection and comparison  
- Proper feature scaling for improved model performance  
- Streamlit interactive web app for real-time predictions  
- Easy-to-use input interface for transaction details  

---

## 🛠 Technologies & Libraries

- Python 3.x  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Pickle (for saving/loading model & scaler)  
