import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Define the paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

st.set_page_config(page_title="Diabetes Prediction", page_icon="🏥", layout="centered")

st.title("🏥 Diabetes Prediction App")
st.write("Enter the patient's medical details below to predict the likelihood of diabetes.")

# Load models and scaler
@st.cache_resource
def load_assets():
    scaler = joblib.load(SCALER_PATH)
    models = {
        "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, 'logistic_regression.pkl')),
        "Random Forest": joblib.load(os.path.join(MODEL_DIR, 'random_forest.pkl')),
        "SVM": joblib.load(os.path.join(MODEL_DIR, 'svm.pkl'))
    }
    return scaler, models

try:
    scaler, models = load_assets()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models. Please ensure they are trained and saved in the 'models' directory. Error: {e}")
    models_loaded = False

if models_loaded:
    st.sidebar.header("Model Selection")
    selected_model_name = st.sidebar.selectbox("Choose a model for prediction", list(models.keys()))
    selected_model = models[selected_model_name]

    st.header("Patient Data Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120, step=1)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70, step=1)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)
        
    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=79, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

    if st.button("Predict", type="primary"):
        input_data = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age
        ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # Scale the data
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = selected_model.predict(input_scaled)[0]
        
        if hasattr(selected_model, "predict_proba"):
            probability = selected_model.predict_proba(input_scaled)[0][1]
            prob_text = f" (Probability: {probability:.2%})"
        else:
            prob_text = ""
            
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"🚨 The model predicts: **Diabetic**{prob_text}")
        else:
            st.success(f"✅ The model predicts: **Non-Diabetic**{prob_text}")
