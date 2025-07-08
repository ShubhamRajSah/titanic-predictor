import streamlit as st
import joblib
import numpy as np
import pandas as pd  # Import pandas

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
    feature_names = [
        "Pclass", "Sex", "Age", "Fare", "Title", "FamilySize", "isAlone", "Embarked", "TicketPrefix", "Deck"
    ]
    input_df = pd.DataFrame([[pclass, 0 if sex == 'male' else 1, age, fare,
                            title, family_size, is_alone, embarked,
                            ticket_prefix, deck]], columns=feature_names)
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    confidence = round(prediction_proba[0][1 if prediction==1 else 0]*100,2)
    if prediction==1:
        st.success(f"🎉 Prediction: Survived")
    else:
        st.error("💀 prediction : Did not Survive")

    st.info(f"🧠 Model Confidence:{confidence}%")   
    
    # part 2 - confidence bar
    st.markdown('## Confidence Meter')
    st.progress(confidence/100)
    
    # show both probabilities
    st.markdown("Prediction Breakdown")
    st.write(f"Not survived: {round(prediction_proba[0][0]*100,2)}")
    st.write(f"Survived: {round(prediction_proba[0][1]*100,2)}")

    # Emoji confidence tag
    if confidence >=85:
        emoji="💪 Very confident "
    elif confidence >= 60:
        emoji ="🙂 Fairly confident "
    else :
        emoji="🤷‍♂️ Not very confident "
    st.markdown(f"**{emoji}**")








