# Importing libraries and dependencies
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model
with open('bank_model.pkl', 'rb') as file:
    model = pickle.load(file)
# # Load the trained model
# model = joblib.load('bank_model.pkl')

# Load the scaler (if you used one for preprocessing)
scaler = joblib.load('scaler.pkl')  # If you saved the scaler during preprocessing

# Define function for prediction
def predict_subscription(input_data):
    # Preprocess the input data (apply the same transformations as during training)
    input_data_scaled = scaler.transform(input_data)
    
    # Predict with the trained model
    prediction = model.predict(input_data_scaled)
    
    return "Yes" if prediction[0] == 1 else "No"


# Title and description
st.title("Bank Term Deposit Subscription Predictor")
st.write("Enter the details of a client to predict whether they will subscribe to a bank term deposit.")


# Collect user input
age = st.number_input('Age', min_value=18, max_value=100, value=30)
job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
marital = st.selectbox('Marital Status', ['single', 'married', 'divorced'])
education = st.selectbox('Education Level', ['primary', 'secondary', 'tertiary', 'unknown'])
balance = st.number_input('Account Balance', value=1000.0)
housing = st.selectbox('Has Housing Loan?', ['yes', 'no'])
loan = st.selectbox('Has Personal Loan?', ['yes', 'no'])
contact = st.selectbox('Contact Communication Type', ['cellular', 'telephone', 'unknown'])
day = st.number_input('Last Contact Day of the Month', min_value=1, max_value=31, value=15)
month = st.selectbox('Last Contact Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
duration = st.number_input('Duration of Last Contact (seconds)', value=150)
campaign = st.number_input('Number of Contacts During This Campaign', value=1)
pdays = st.number_input('Number of Days After Last Contact', value=-1)
previous = st.number_input('Number of Contacts Before This Campaign', value=0)
poutcome = st.selectbox('Outcome of Previous Marketing Campaign', ['failure', 'nonexistent', 'success'])


# Convert categorical inputs to the same format used in training (one-hot encoding)
input_data = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'balance': [balance],
    'housing': [housing],
    'loan': [loan],
    'contact': [contact],
    'day': [day],
    'month': [month],
    'duration': [duration],
    'campaign': [campaign],
    'pdays': [pdays],
    'previous': [previous],
    'poutcome': [poutcome]
})

# Apply one-hot encoding for categorical features (if needed)
input_data = pd.get_dummies(input_data, drop_first=True)

# Ensure the input data has the same features as the training data
input_data = input_data.reindex(columns=model.feature_importances_.index, fill_value=0)


# Prediction button
if st.button('Predict Subscription'):
    prediction = predict_subscription(input_data)
    st.write(f"The predicted outcome is: {prediction}")
