import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Diabetes Prediction System", layout="centered")
st.title("🩺 Diabetes Prediction System")

# ---------------- LOAD MODEL & PREPROCESSORS ----------------
model = joblib.load("models/best_model.pkl")
imputer = joblib.load("models/imputer.pkl")
scaler = joblib.load("models/scaler.pkl")
selector = joblib.load("models/selector.pkl")

# ---------------- USER INPUTS ----------------
Pregnancies = st.number_input("Pregnancies", 0, 20)
Glucose = st.number_input("Glucose", 0, 300)
BloodPressure = st.number_input("Blood Pressure", 0, 150)
SkinThickness = st.number_input("Skin Thickness", 0, 100)
Insulin = st.number_input("Insulin", 0, 900)
BMI = st.number_input("BMI", 0.0, 70.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
Age = st.number_input("Age", 1, 120)
Smoke = st.selectbox(
    "Do you smoke?",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    # Create DataFrame (same order as training)
    X = pd.DataFrame([[
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age,
        Smoke
    ]], columns=[
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
        "Smoke"
    ])

    # Preprocessing
    X = imputer.transform(X)
    X = scaler.transform(X)
    X = selector.transform(X)

    # ---------------- ML PROBABILITY ----------------
    ml_prob = model.predict_proba(X)[0][1]

    # ---------------- RULE-BASED RISK ADJUSTMENT ----------------
    risk_score = 0.0
    reasons = []

    if Glucose >= 180:
        risk_score += 0.15
        reasons.append("High Glucose Level")

    if BMI >= 35:
        risk_score += 0.15
        reasons.append("High BMI (Obesity)")

    if Age >= 50:
        risk_score += 0.10
        reasons.append("Age Above 50")

    if DiabetesPedigreeFunction >= 1.0:
        risk_score += 0.15
        reasons.append("Strong Family History of Diabetes")

    if Insulin >= 250:
        risk_score += 0.10
        reasons.append("Abnormal Insulin Level")

    if Smoke == 1:
        risk_score += 0.10
        reasons.append("Smoking Habit")

    # Final probability
    final_prob = min(ml_prob + risk_score, 1.0)

    # ---------------- DISPLAY OUTPUT ----------------
    st.subheader(f"Diabetes Probability: {final_prob:.2f}")

    if final_prob >= 0.5:
        st.error("⚠️ High Risk of Diabetes")

        st.markdown("### 🧠 Reasons for High Risk:")
        for r in reasons:
            st.write(f"• {r}")

    else:
        st.success("✅ Low Risk of Diabetes")

        if reasons:
            st.markdown("### ⚠️ Potential Risk Factors Observed:")
            for r in reasons:
                st.write(f"• {r}")
