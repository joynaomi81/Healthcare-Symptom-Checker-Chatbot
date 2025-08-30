import streamlit as st
import pandas as pd
import joblib

# --- Load trained model and column order ---
model = joblib.load("healthcare_model.pkl")
model_columns = joblib.load("model_columns.pkl")  

st.title("🩺 Conversational Healthcare Symptom Checker Chatbot")
st.write("Answer the questions below about your symptoms and health info.")

# --- Initialize session state ---
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.responses = {}

# --- Mapping dictionaries ---
binary_map = {"Yes":1, "No":0}
gender_map = {"Male":0, "Female":1}
bp_map = {"Normal":0, "High":1}
chol_map = {"Normal":0, "High":1}

# --- List of questions ---
questions = [
    ("Fever", ["Yes", "No"]),
    ("Cough", ["Yes", "No"]),
    ("Fatigue", ["Yes", "No"]),
    ("Difficulty Breathing", ["Yes", "No"]),
    ("Age", None),  # Numerical input
    ("Gender", ["Male", "Female"]),
    ("Blood Pressure", ["Normal", "High"]),
    ("Cholesterol Level", ["Normal", "High"]),
    ("Disease", None)  
]

# --- Conversation flow ---
if st.session_state.step < len(questions):
    q, options = questions[st.session_state.step]

    if options:  # Radio button
        response = st.radio(f"{q}?", options, key=q)
    else:  # Text input / number input
        if q == "Age":
            response = st.number_input(f"{q}:", min_value=0, max_value=120, key=q)
        else:
            response = st.text_input(f"{q}:", key=q)

    if st.button("Next"):
        st.session_state.responses[q] = response
        st.session_state.step += 1
        st.experimental_rerun()

else:
    st.write("All questions answered. Checking outcome...")
    
    # --- Preprocess input ---
    input_data = pd.DataFrame({
        'Fever': [binary_map.get(st.session_state.responses.get("Fever"),0)],
        'Cough': [binary_map.get(st.session_state.responses.get("Cough"),0)],
        'Fatigue': [binary_map.get(st.session_state.responses.get("Fatigue"),0)],
        'Difficulty Breathing': [binary_map.get(st.session_state.responses.get("Difficulty Breathing"),0)],
        'Age': [st.session_state.responses.get("Age",0)],
        'Gender': [gender_map.get(st.session_state.responses.get("Gender"),0)],
        'Blood Pressure': [bp_map.get(st.session_state.responses.get("Blood Pressure"),0)],
        'Cholesterol Level': [chol_map.get(st.session_state.responses.get("Cholesterol Level"),0)],
        'Disease': [0] 
    })

    # Align columns with training data
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][prediction]
    result = "Positive" if prediction==1 else "Negative"

    st.success(f"Predicted Outcome: {result}")
    st.info(f"Prediction Confidence: {proba*100:.2f}%")

    # Reset button
    if st.button("Restart"):
        st.session_state.step = 0
        st.session_state.responses = {}
        st.experimental_rerun()
