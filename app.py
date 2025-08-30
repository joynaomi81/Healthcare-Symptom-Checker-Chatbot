import streamlit as st
import pandas as pd
import joblib

# --- Load trained model and column order ---
model = joblib.load("healthcare_model.pkl")
model_columns = joblib.load("model_columns.pkl")  
st.title("🩺 Healthcare Symptom Checker Chatbot")
st.write("Enter your symptoms and health information to check your outcome.")

# --- Collect user input ---
Fever = st.radio("Do you have Fever?", ["Yes", "No"])
Cough = st.radio("Do you have Cough?", ["Yes", "No"])
Fatigue = st.radio("Do you have Fatigue?", ["Yes", "No"])
Difficulty = st.radio("Difficulty Breathing?", ["Yes", "No"])
Age = st.number_input("Age", min_value=0, max_value=120)
Gender = st.radio("Gender", ["Male", "Female"])
BP = st.radio("Blood Pressure", ["Normal", "High"])
Chol = st.radio("Cholesterol Level", ["Normal", "High"])
Disease = st.text_input("Disease")  
# --- Preprocess input ---
binary_map = {"Yes":1, "No":0}
gender_map = {"Male":0, "Female":1}
bp_map = {"Normal":0, "High":1}
chol_map = {"Normal":0, "High":1}


input_data = pd.DataFrame({
    'Fever': [binary_map[Fever]],
    'Cough': [binary_map[Cough]],
    'Fatigue': [binary_map[Fatigue]],
    'Difficulty Breathing': [binary_map[Difficulty]],
    'Age': [Age],
    'Gender': [gender_map[Gender]],
    'Blood Pressure': [bp_map[BP]],
    'Cholesterol Level': [chol_map[Chol]],
    'Disease': [0]  
})

input_data = input_data.reindex(columns=model_columns, fill_value=0)


if st.button("Check Outcome"):
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][prediction]
    result = "Positive" if prediction==1 else "Negative"
    st.success(f"Predicted Outcome: {result}")
    st.info(f"Prediction Confidence: {proba*100:.2f}%")
