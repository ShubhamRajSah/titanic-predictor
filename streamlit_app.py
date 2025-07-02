import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('final_random_forest_model.pkl')

st.title("🎯 Titanic Survival Predictor")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ['male', 'female'])
age = st.slider("Age", 0, 80, 25)
fare = st.slider("Fare", 0.0, 500.0, 50.0)
title = st.selectbox("Title", [0, 1, 2])  # Adjust based on label encoding
family_size = st.slider("Family Size", 1, 11, 1)
is_alone = 1 if family_size == 1 else 0
embarked = st.selectbox("Embarked", [0, 1, 2])
ticket_prefix = st.selectbox("Ticket Prefix", [0, 1, 2])  # Adjust as needed
deck = st.selectbox("Deck", [0, 1, 2, 3, 4, 5, 6, 7])  # Adjust mapping

# Prediction
if st.button("Predict"):
    input_data = np.array([[pclass, 0 if sex == 'male' else 1, age, fare,
                            title, family_size, is_alone, embarked,
                            ticket_prefix, deck]])
    prediction = model.predict(input_data)[0]
    st.success(f"🎉 Prediction: {'Survived' if prediction == 1 else 'Did Not Survive'}")
    





    