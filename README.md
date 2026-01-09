🩺**Diabetes_Prediction_using_Classification_and_Ensemble_Learning_Technique**
 Diabetes Prediction System  This project presents an end-to-end diabetes risk prediction system using machine learning ensemble techniques combined with rule-based explainability. The system predicts the probability of diabetes based on user health parameters and explains the key contributing risk factors behind each prediction.

# 🩺 Diabetes Prediction System (Major Project)

An **Explainable Hybrid Machine Learning–based Diabetes Risk Prediction System** developed as a **major academic project**.  
This system predicts the probability of diabetes using ensemble machine learning techniques and provides **clear reasons behind each prediction**, making it interpretable and user-friendly.

---

## 📌 Project Motivation

Diabetes is a chronic disease that requires early risk assessment and awareness.  
Traditional machine learning models often act as black boxes, making it difficult for users to understand predictions.

This project addresses that limitation by combining:
- Machine Learning predictions
- Rule-based explainability (Explainable AI)

---

## 🎯 Objectives

- To build an accurate diabetes risk prediction model
- To apply ensemble learning for improved performance
- To make predictions **interpretable** by showing risk factors
- To deploy the model using a user-friendly web interface

---

## 🚀 Key Features

- ✅ Balanced dataset (50% diabetic, 50% non-diabetic)
- ✅ Ensemble learning using **Stacking Classifier**
- ✅ Models used:
  - Logistic Regression  
  - Decision Tree  
  - Support Vector Machine (SVM)  
  - Random Forest  
  - Gradient Boosting  
- ✅ Hybrid prediction logic (ML probability + rule-based risk scoring)
- ✅ Explainable AI: displays reasons for high-risk predictions
- ✅ Evaluation using Accuracy, Precision, Recall, F1-score, and **ROC–AUC**
- ✅ Web deployment using **Streamlit**

---

## 🧠 System Architecture & Workflow

1. Data preprocessing (imputation, scaling, feature selection)
2. Train–test split on balanced dataset
3. Ensemble model training using stacking
4. Model evaluation using multiple metrics
5. Real-time prediction and explanation via Streamlit app

---

## 🧩 Explainable AI Component

For **high-risk predictions**, the system explains the reasons such as:
- High glucose level
- High BMI (obesity)
- Advanced age
- Family history of diabetes
- Smoking habit
- Abnormal insulin level

This improves **transparency and trust** in the system.

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Machine Learning:** Scikit-learn  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib  
- **Web Application:** Streamlit  
- **Model Persistence:** Joblib  
- **Version Control:** Git & GitHub  

---

## 📂 Project Structure

diabetes_prediction_system/
│
├── data/
│ └── diabetes_balanced_20k_50_50.csv
│
├── models/
│ ├── best_model.pkl
│ ├── imputer.pkl
│ ├── scaler.pkl
│ └── selector.pkl
│
├── scripts/
│ ├── data_prep.py
│ ├── train.py
│ └── evaluate.py
│
├── app/
│ └── app.py
│
├── images/
│ ├── non_diabetic_output.png
│ └── diabetic_output.png
│
├── requirements.txt
└── README.md



---

## ▶️ How to Run the Project

bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts/data_prep.py
python scripts/train.py
python scripts/evaluate.py
python -m streamlit run app/app.py


**📊 Model Performance**

High accuracy on balanced dataset
ROC–AUC score ≈ 0.90
Strong class discrimination capability

**🖼️ Sample Outputs**
🟢 Non-Diabetic Prediction Output

Description:
The system predicts a low probability of diabetes with no major risk factors detected.

**Input Parameters**
Pregnancies: 0
Glucose: 90 mg/dL
Blood Pressure: 70 mmHg
Skin Thickness: 20 mm
Insulin: 80 µU/mL
BMI: 22.00
Diabetes Pedigree Function: 0.25
Age: 25 years
Smoking Status: No

**Model Output**
Diabetes Probability: 0.13
Prediction: ✅ Low Risk of Diabete

🔴 Diabetic Prediction Output

Description:
The system predicts a high probability of diabetes and clearly displays contributing risk factors such as high glucose level, obesity (BMI), age, family history, and smoking habit.

**Input Parameters**
Pregnancies: 5
Glucose: 170 mg/dL
Blood Pressure: 92 mmHg
Skin Thickness: 38 mm
Insulin: 220 µU/mL
BMI: 34.00
Diabetes Pedigree Function: 1.20
Age: 55 years
Smoking Status: Yes

**Model Output**
Diabetes Probability: 0.53

Prediction: ⚠️ High Risk of Diabetes
Reasons for High-Risk Prediction
The system identifies a high risk of diabetes due to the following contributing factors:
Age above 50, which increases diabetes risk
Strong family history of diabetes (high Diabetes Pedigree Function)
Smoking habit, which is a known risk factor
Elevated glucose level and increased BMI further support the high-risk classificatio
