import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib

# ==========================================
# Theme & Page Config
# ==========================================
st.set_page_config(
    page_title="Cardiovascular Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Cardiovascular Disease Risk Prediction App")
st.write("Enter the patient's clinical parameters below to evaluate cardiovascular risk.")

# ==========================================
# Model Loading with Caching
# ==========================================
@st.cache_resource
def load_prediction_models():
    try:
        # 1. CatBoost Classifier
        cat_m = CatBoostClassifier()
        cat_m.load_model("catboost_model.json", format="json")
        
        # 2. XGBoost Classifier 
        xgb_m = xgb.XGBClassifier()
        xgb_m.load_model("xgboost_model.json")
        
        # 3. LightGBM Classifier 
        lgb_m = joblib.load("lightgbm_model.pkl") 
        
        return cat_m, xgb_m, lgb_m
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.warning("Make sure your trained model files are in the same folder as app.py!")
        return None, None, None

cat_model, xgb_model, lgbm_model = load_prediction_models()

# ==========================================
# User Interface (Inputs)
# ==========================================
st.subheader("📋 Patient Clinical Profiles")

col1, col2 = st.columns(2)

with col1:
    age_years = st.number_input("Age (Years)", min_value=1, max_value=120, value=25)
    
    gender_selection = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender_selection == "Male" else 0
    
    st.write("**Height Assignment**")
    height_ft = st.number_input("Feet", min_value=1, max_value=8, value=5)
    height_in = st.number_input("Inches", min_value=0, max_value=11, value=6)
    
    weight_kg = st.number_input("Weight (kg)", min_value=10.0, max_value=250.0, value=75.0, step=0.5)

with col2:
    cholesterol = st.number_input("Cholesterol Level (mg/dL)", min_value=100, max_value=500, value=170)
    systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=250, value=115)
    diastolic_bp = st.number_input("Diastolic Blood Pressure (ap_lo)", min_value=40, max_value=150, value=75)

st.subheader("🩺 Lifestyle & Metabolic Parameters")
col3, col4 = st.columns(2)

with col3:
    glucose_sel = st.selectbox("Glucose Level", ["Normal", "Above Normal", "Well Above Normal"])
    gluc = 1 if glucose_sel == "Normal" else (2 if glucose_sel == "Above Normal" else 3)
    
    smoke_sel = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker"])
    smoke = 1 if smoke_sel == "Smoker" else 0

with col4:
    alco_sel = st.selectbox("Alcohol Intake", ["No", "Yes"])
    alco = 1 if alco_sel == "Yes" else 0
    
    active_sel = st.selectbox("Physical Activity", ["Active", "Inactive"])
    active = 1 if active_sel == "Active" else 0

# ==========================================
# Feature Engineering & Live Calculations
# ==========================================
# 1. Height conversion (Feet & Inches to cm and meters)
total_inches = (height_ft * 12) + height_in
height_cm = total_inches * 2.54
height_meters = total_inches * 0.0254

# 2. Live BMI Calculation
if height_meters > 0:
    calculated_bmi = weight_kg / (height_meters ** 2)
else:
    calculated_bmi = 0.0

# 3. Live Pulse Pressure Calculation
calculated_pulse_pressure = systolic_bp - diastolic_bp

# 4. Live Age-BMI Risk Calculation (Age in Years)
calculated_age_bmi_risk = age_years * calculated_bmi

# 5. Cholesterol Mapping (mg/dL to 1, 2, 3 categories)
if cholesterol < 200:
    cholesterol_mapped = 1  # Normal
elif 200 <= cholesterol < 240:
    cholesterol_mapped = 2  # Above Normal
else:
    cholesterol_mapped = 3  # Well Above Normal

# Display Live BMI
st.info(f"**Calculated BMI:** {calculated_bmi:.2f} kg/m²")

# ==========================================
# Master Dictionary
# ==========================================
input_data = {
    'age': [age_years],
    'gender': [gender],
    'height': [height_cm],
    'weight': [weight_kg],
    'ap_hi': [systolic_bp],
    'ap_lo': [diastolic_bp],
    'cholesterol': [cholesterol_mapped],
    'gluc': [gluc],            
    'smoke': [smoke],           
    'alco': [alco],            
    'active': [active],          
    'bmi': [calculated_bmi],
    'pulse_pressure': [calculated_pulse_pressure],
    'age_bmi_risk': [calculated_age_bmi_risk]
}
X_test_case = pd.DataFrame(input_data)

# Reorder columns to match CatBoost model feature order
if cat_model is not None and hasattr(cat_model, 'feature_names_'):
    X_test_case = X_test_case[cat_model.feature_names_]

# ==========================================
# Prediction 50:30:20 Weighted Fusion
# ==========================================
st.markdown("---")

if st.button("Analyze Cardiovascular Risk", type="primary"):
    if cat_model is not None and xgb_model is not None and lgbm_model is not None:
        with st.spinner("Processing framework matrices..."):
            prob_cat = cat_model.predict_proba(X_test_case)[0, 1]
            prob_xgb = xgb_model.predict_proba(X_test_case)[0, 1]
            prob_lgbm = lgbm_model.predict_proba(X_test_case)[0, 1]
            
            final_probability = (0.50 * prob_cat) + (0.30 * prob_xgb) + (0.20 * prob_lgbm)
            
            st.subheader("Diagnostic Output Analysis")
            
            if final_probability >= 0.5:
                st.error(f"**Cardiovascular Disease Detected (High Risk)**")
            else:
                st.success(f"✅ **Healthy / Low Risk Configuration**")
                
            st.write(f"**Ensemble-Weighted Risk Probability:** {final_probability * 100:.2f}%")
            st.progress(float(final_probability))
    else:
        st.error("Cannot perform analysis. Model files are missing or could not be loaded.")
