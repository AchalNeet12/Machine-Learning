import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Specify the full paths to the pickle files
model_path = r"E:\3.spyder\Clasification\logistic regression\logistic_regression_model.pkl"
scaler_path = r"E:\3.spyder\Clasification\logistic regression\scaler.pkl"

# Load the trained model and scaler
classifier = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Streamlit App
st.title("Logistic Regression Prediction App")

st.write("""
This app predicts the outcome using a logistic regression model.
Enter the features below to get the prediction.
""")

# Input fields
feature1 = st.number_input("Enter Feature 1", step=0.01)
feature2 = st.number_input("Enter Feature 2", step=0.01)

# Prediction button
if st.button("Predict"):
    try:
        # Preprocess input
        data = np.array([[feature1, feature2]])
        data_scaled = scaler.transform(data)

        # Make prediction
        prediction = classifier.predict(data_scaled)
        result = "Positive" if prediction[0] == 1 else "Negative"

        # Display result
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
