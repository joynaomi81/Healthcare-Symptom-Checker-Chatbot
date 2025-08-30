import streamlit as st
import pandas as pd
import joblib

# --- Load model and column order ---
model = joblib.load("healthcare_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🩺 Healthcare Symptom Checker Chatbot")
st.write("Answer the questions to check your health outcome.\n")

# --- Initialize session state ---
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.responses = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
    ("Age", None),
    ("Gender", ["Male", "Female"]),
    ("Blood Pressure", ["Normal", "High"]),
    ("Cholesterol Level", ["Normal", "High"]),
    ("Disease", None)
]

# --- Helper to display chat bubbles ---
def display_chat():
    for speaker, message in st.session_state.chat_history:
        if speaker == "bot":
            st.markdown(f"""
            <div style='text-align: left; margin:10px 0;'>
                <span style='background-color:#F0F0F0; padding:10px; border-radius:10px; display:inline-block;'>{message}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='text-align: right; margin:10px 0;'>
                <span style='background-color:#A0E7E5; padding:10px; border-radius:10px; display:inline-block;'>{message}</span>
            </div>
            """, unsafe_allow_html=True)

# --- Display previous chat ---
display_chat()

# --- Conversation flow ---
if st.session_state.step < len(questions):
    q, options = questions[st.session_state.step]

    # Show bot question
    if len(st.session_state.chat_history) == st.session_state.step*2:
        st.session_state.chat_history.append(("bot", q + "?"))
        display_chat()

    # Get user input
    if options:
        response = st.radio(f"{q}?", options, key=f"{q}_radio")
    else:
        if q == "Age":
            response = st.number_input(f"{q}:", min_value=0, max_value=120, key=f"{q}_num")
        else:
            response = st.text_input(f"{q}:", key=f"{q}_text")

    next_clicked = st.button("Send")
    if next_clicked:
        # Store user response
        st.session_state.responses[q] = response
        st.session_state.chat_history.append(("user", str(response)))
        st.session_state.step += 1
        st.experimental_rerun()

else:
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
        'Disease': [0]  # Replace with proper encoding if needed
    })

    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][prediction]
    result = "Positive" if prediction == 1 else "Negative"

    st.session_state.chat_history.append(("bot", f"Predicted Outcome: {result}"))
    st.session_state.chat_history.append(("bot", f"Confidence: {proba*100:.2f}%"))
    display_chat()

    if st.button("Restart"):
        st.session_state.step = 0
        st.session_state.responses = {}
        st.session_state.chat_history = []
        st.experimental_rerun()
