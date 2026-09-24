import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Diabetes Prediction AI", page_icon="🩺",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.main {background-color:#f8fafc;}
.block-container {padding-top:2rem;padding-bottom:3rem;max-width:1200px;}
.hero-title {font-size:44px;font-weight:800;margin-bottom:5px;}
.hero-subtitle {font-size:18px;color:#64748b;margin-bottom:25px;}
.card {background:white;padding:24px;border-radius:18px;border:1px solid #e2e8f0;
       box-shadow:0 8px 25px rgba(15,23,42,.06);margin-bottom:20px;}
.footer{text-align:center;color:#64748b;padding-top:30px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    package = joblib.load("models/diabetes_model.pkl")
    return package

st.markdown('<div class="hero-title">🩺 Diabetes Prediction AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Machine Learning powered diabetes risk assessment</div>', unsafe_allow_html=True)
st.info("Educational machine-learning demonstration only — not a medical diagnosis. Do not use this result to make treatment decisions.")

try:
    package = load_model()
    model = package["model"]
    scaler = package["scaler"]
    model_name = package["model_name"]
    features = package["features"]
except Exception as e:
    st.error("Model file not found or could not be loaded. Run diabetes.ipynb and ensure it saves models/diabetes_model.pkl.")
    st.stop()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("👤 Patient Information")
c1, c2, c3 = st.columns(3)

with c1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120, step=1,
                              help="Plasma glucose concentration.")
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70, step=1)

with c2:
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)
    insulin = st.number_input("Insulin (µU/mL)", min_value=0, max_value=900, value=80, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0, step=0.1)

with c3:
    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0,
                               value=0.5, step=0.01)
    age = st.number_input("Age (years)", min_value=1, max_value=120, value= 30, step=1)

st.markdown("</div>", unsafe_allow_html=True)
st.caption("In this dataset, zero values for Glucose, Blood Pressure, Skin Thickness, Insulin, or BMI may represent missing/invalid measurements. Interpret inputs carefully.")

if st.button("🔍 ANALYZE DIABETES RISK", use_container_width=True, type="primary"):
    values = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age
    }
    input_data = pd.DataFrame([[values[f] for f in features]], columns=features)
    model_input = scaler.transform(input_data) if model_name in ["KNN", "Logistic Regression"] else input_data
    prediction = int(model.predict(model_input)[0])
    probability = float(model.predict_proba(model_input)[0][1])

    st.markdown("---")
    st.subheader("📊 Prediction Result")
    left, right = st.columns(2)
    with left:
        if prediction == 1:
            st.warning("⚠️ Model predicts a diabetes-positive outcome")
        else:
            st.success("✅ Model predicts a diabetes-negative outcome")
    with right:
        st.metric("Estimated Positive-Class Probability", f"{probability * 100:.1f}%")

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100,
        title={"text": "Model-estimated probability"},
        gauge={"axis": {"range": [0, 100]},
               "steps": [{"range": [0, 40]}, {"range": [40, 70]}, {"range": [70, 100]}]}
    ))
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This probability reflects the trained model and dataset, not a clinically validated individual risk estimate.")

st.markdown("---")
st.subheader("🤖 Model Information")
a, b, c = st.columns(3)
a.metric("Algorithm", model_name)
b.metric("Features", len(features))
c.metric("Learning Type", "Supervised ML")
st.markdown('<div class="footer">Diabetes Prediction AI • Machine Learning Project</div>', unsafe_allow_html=True)
