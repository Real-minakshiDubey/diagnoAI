<div align="center">

# 🏥 DiagnoAI
### Multi Disease Prediction System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://diagnoai-asnd5d6rqcnytnpdv6yjsb.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

**An AI-powered clinical screening tool that predicts Breast Cancer, Diabetes, and Heart Disease using Machine Learning**

[🚀 Live Demo](https://diagnoai-asnd5d6rqcnytnpdv6yjsb.streamlit.app/) • [📊 Datasets](#-datasets) • [🛠️ Tech Stack](#-tech-stack) • [📁 Project Structure](#project-structure)

</div>

---

## 📌 Overview

**DiagnoAI** is an end-to-end machine learning web application built as part of an AIML BTech project. It allows medical professionals to input patient diagnostic data and receive instant AI-powered disease predictions with confidence scores and clinical recommendations.

The system covers **3 critical diseases**:
- 🎗️ **Breast Cancer** — Based on Fine Needle Aspiration (FNA) biopsy measurements
- 🩺 **Diabetes** — Based on glucose levels, BMI, insulin, and other metabolic indicators  
- ❤️ **Heart Disease** — Based on ECG results, cholesterol, chest pain type, and cardiovascular metrics

---

## 🚀 Live Demo

> 🌐 **[https://diagnoai-asnd5d6rqcnytnpdv6yjsb.streamlit.app/](https://diagnoai-asnd5d6rqcnytnpdv6yjsb.streamlit.app/)**

---

## ✨ Features

- ✅ Predicts 3 diseases from a single unified interface
- ✅ Shows prediction confidence with a visual progress bar
- ✅ Provides clinical recommendations based on result
- ✅ Tabbed input layout for clean data entry
- ✅ Tooltips on every input field explaining medical terms
- ✅ Fully responsive professional UI with custom CSS
- ✅ Medical disclaimer for responsible AI use

---

## 📊 Model Performance

| Disease | Algorithm | Accuracy | F1 Score | ROC-AUC |
|---------|-----------|----------|----------|---------|
| 🎗️ Breast Cancer | Logistic Regression | 97.37% | 0.964 | 0.994 |
| 🩺 Diabetes | Random Forest | 76.62% | 0.679 | 0.840 |
| ❤️ Heart Disease | Random Forest | 86.96% | 0.883 | 0.933 |

---

## 🔬 Datasets

| Disease | Dataset | Source | Rows | Features |
|---------|---------|--------|------|----------|
| Breast Cancer | Wisconsin Diagnostic Dataset | Kaggle | 569 | 32 |
| Diabetes | PIMA Indians Diabetes Dataset | Kaggle | 768 | 9 |
| Heart Disease | Heart Failure Prediction Dataset | Kaggle | 918 | 12 |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10 |
| **ML Libraries** | Scikit-learn, XGBoost |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Streamlit |
| **Deployment** | Streamlit Cloud |
| **Version Control** | Git & GitHub |

---

## 🧠 ML Pipeline

```
Raw Data
   ↓
Exploratory Data Analysis (EDA)
   ↓
Data Preprocessing
(Missing values, Encoding, Scaling)
   ↓
Model Training
(Logistic Regression, Random Forest, SVM, XGBoost)
   ↓
Model Evaluation
(Accuracy, F1 Score, ROC-AUC, Confusion Matrix)
   ↓
Best Model Selection
   ↓
Streamlit Web App
   ↓
Deployment on Streamlit Cloud
```

---

## 📁 Project Structure

```
diagnoAI/
│
├── models/
│   ├── cancer_model.pkl       # Trained Logistic Regression
│   ├── cancer_scaler.pkl      # StandardScaler for cancer
│   ├── diabetes_model.pkl     # Trained Random Forest
│   ├── diabetes_scaler.pkl    # StandardScaler for diabetes
│   ├── heart_model.pkl        # Trained Random Forest
│   └── heart_scaler.pkl       # StandardScaler for heart
│
├── notebooks/
│   ├── cancer_analysis.ipynb  # EDA + Preprocessing + Model
│   ├── diabetes_analysis.ipynb
│   └── heart_analysis.ipynb
│
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🔍 Key EDA Findings

### Breast Cancer
- Dataset: 569 patients, 32 features
- Best predictors: `concave_points_mean` (0.8 correlation), `perimeter_worst` (0.8)
- Class balance: 63% Benign, 37% Malignant

### Diabetes
- Dataset: 768 patients, 9 features  
- Hidden zeros found in 5 columns (replaced with median)
- Best predictor: `Glucose` (0.47 correlation with outcome)
- Class balance: 65% Non-diabetic, 35% Diabetic

### Heart Disease
- Dataset: 918 patients, 12 features
- 5 categorical columns encoded (Label + One Hot Encoding)
- Best predictors: `ST_Slope_Flat` (0.55), `ChestPainType_ASY` (0.52)
- Best balanced dataset: 55% Disease, 45% No Disease

---

## ⚙️ Run Locally

```bash
# Clone the repository
git clone https://github.com/Real-minakshiDubey/diagnoAI.git

# Navigate to project
cd diagnoAI

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## ⚠️ Medical Disclaimer

> DiagnoAI is designed as an **educational and supportive clinical screening tool**.
> It does **not** replace professional medical advice, diagnosis, or treatment.
> Always consult a qualified healthcare professional for medical decisions.

---

## 👩‍💻 Author

**Minakshi Dubey**
B.Tech — Artificial Intelligence & Machine Learning


[![GitHub](https://img.shields.io/badge/GitHub-Real--minakshiDubey-black?logo=github)](https://github.com/Real-minakshiDubey)

---

<div align="center">
⭐ If you found this project helpful, please give it a star!
</div>
