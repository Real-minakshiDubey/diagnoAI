# ===== IMPORTS =====
import streamlit as st
import numpy as np
import joblib

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="DiagnoAI — Multi Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# ===== LOAD MODELS & SCALERS =====
@st.cache_resource
def load_models():
    models = {
        'cancer': joblib.load('models/cancer_model.pkl'),
        'diabetes': joblib.load('models/diabetes_model.pkl'),
        'heart': joblib.load('models/heart_model.pkl')
    }
    scalers = {
        'cancer': joblib.load('models/cancer_scaler.pkl'),
        'diabetes': joblib.load('models/diabetes_scaler.pkl'),
        'heart': joblib.load('models/heart_scaler.pkl')
    }
    return models, scalers

models, scalers = load_models()

# ===== CUSTOM CSS INJECTION =====
st.markdown(
    """
    <style>
    /* Custom Styling for DiagnoAI */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Apply Font globally */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Center the main header and give it a gradient look */
    .hero-container {
        text-align: center;
        padding: 2.5rem 1.5rem;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.7) 0%, rgba(30, 41, 59, 0.4) 100%);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
    }
    .hero-title {
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.2rem;
        margin-top: 10px;
        margin-bottom: 0;
        font-weight: 350;
    }
    
    /* Styled container cards for sections */
    .section-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Custom button styling */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 100%);
        color: white !important;
        border: none !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        font-size: 1.15rem !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-top: 15px;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #38BDF8 0%, #4F46E5 100%) !important;
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    div.stButton > button:active {
        transform: translateY(1px) !important;
    }
    
    /* Result Cards */
    .result-card {
        border-radius: 14px;
        padding: 24px;
        margin-top: 20px;
        color: white;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
    }
    .result-card-positive {
        background: linear-gradient(135deg, #7F1D1D 0%, #991B1B 50%, #B91C1C 100%);
        border-left: 6px solid #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    .result-card-negative {
        background: linear-gradient(135deg, #064E3B 0%, #065F46 50%, #047857 100%);
        border-left: 6px solid #10B981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .result-title {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .result-metric {
        font-size: 1.15rem;
        opacity: 0.95;
        margin-bottom: 16px;
        font-weight: 300;
    }
    .progress-container {
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 9999px;
        height: 12px;
        overflow: hidden;
        margin-bottom: 18px;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 9999px;
        background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 100%);
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    }
    .result-tips {
        font-size: 0.95rem;
        background: rgba(255, 255, 255, 0.08);
        padding: 14px 20px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        line-height: 1.6;
    }
    
    /* Stylize tabs in Streamlit */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 12px 24px !important;
        color: #94A3B8 !important;
        font-weight: 500 !important;
        font-size: 1.05rem !important;
        transition: all 0.2s ease !important;
        border: 1px solid #334155 !important;
        border-bottom: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #E2E8F0 !important;
        background-color: rgba(30, 41, 59, 0.8) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0EA5E9 !important;
        color: white !important;
        border-color: #0EA5E9 !important;
        font-weight: 600 !important;
    }
    
    /* Sidebar styling overrides */
    section[data-testid="stSidebar"] {
        border-right: 1px solid #334155 !important;
    }

    /* Custom footer styles */
    .footer-container {
        text-align: center;
        padding: 1.5rem;
        margin-top: 3rem;
        border-top: 1px solid #334155;
        color: #64748B;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===== SIDEBAR =====
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 25px; padding-top: 10px;">
        <span style="font-size: 3.5rem;">🏥</span>
        <h2 style="margin: 10px 0 0 0; color: #0EA5E9; font-weight: 800; font-size: 1.8rem; letter-spacing: -0.5px;">DiagnoAI</h2>
        <p style="color: #94A3B8; font-size: 0.9rem; margin-top: 4px; font-weight: 300;">Multi Disease Diagnostics</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")

# Disease Selector (styled sidebar dropdown)
st.sidebar.markdown("### 🎯 Diagnostic Target")
disease = st.sidebar.selectbox(
    "Select disease classifier model:",
    ["Breast Cancer", "Diabetes", "Heart Disease"],
    help="Select which machine learning classification model to load and use for prediction."
)

st.sidebar.markdown("---")

# Diagnostics Status Card
st.sidebar.markdown(
    """
    <div style="background-color: #1E293B; border: 1px solid #334155; padding: 18px; border-radius: 10px; margin-top: 10px;">
        <h4 style="margin: 0 0 12px 0; color: #0EA5E9; font-size: 0.95rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">🧠 Active Models</h4>
        <div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.8;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span>🎗️ Breast Cancer</span>
                <span style="color: #10B981; font-weight: 500;">Logistic Reg.</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span>🩺 Diabetes</span>
                <span style="color: #10B981; font-weight: 500;">Logistic Reg.</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>❤️ Heart Disease</span>
                <span style="color: #10B981; font-weight: 500;">Random Forest</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("""
---
### 📊 Model Performance
| Disease | Accuracy |
|---------|----------|
| Cancer | 97.37% |
| Diabetes | 76.62% |
| Heart | 86.96% |
""")

st.sidebar.markdown("""
---
### ℹ️ About DiagnoAI
Built with Python, Scikit-learn
& Streamlit by Minakshi Dubey🎓
""")

# ===== HERO HEADER SECTION =====
st.markdown(
    """
    <div class="hero-container">
        <h1 class="hero-title">🏥 DiagnoAI</h1>
        <p class="hero-subtitle">Multi Disease Prediction System powered by AI</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ================================================
# ===== BREAST CANCER PREDICTION =====
# ================================================
if disease == "Breast Cancer":
    st.markdown("### 🎗️ Breast Cancer Prediction")
    st.info("ℹ️ Values are obtained from FNA biopsy lab reports. Intended for medical professionals.")
    st.markdown("Enter the fine-needle aspirate (FNA) cell nucleus measurements categorized below:")
    
    # Categorize breast cancer input using tabs
    tab1, tab2, tab3 = st.tabs([
        "📏 Mean Dimensions", 
        "🔬 Standard Error Measures", 
        "⚠️ Worst Cell Measurements"
    ])
    
    with tab1:
        st.markdown("<p style='color:#94A3B8; font-size:0.95rem;'>Nuclear characteristics computed from each cell nucleus image (Averages):</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            radius_mean = st.number_input("Radius Mean", 0.0, 30.0, 14.0, help="Mean of distances from center to points on the perimeter")
            texture_mean = st.number_input("Texture Mean", 0.0, 40.0, 19.0, help="Standard deviation of gray-scale values")
            perimeter_mean = st.number_input("Perimeter Mean", 0.0, 200.0, 92.0, help="Mean perimeter size of cell nuclei")
            area_mean = st.number_input("Area Mean", 0.0, 2500.0, 655.0, help="Mean area size of cell nuclei")
            smoothness_mean = st.number_input("Smoothness Mean", 0.0, 0.2, 0.096, format="%.4f", help="Mean of local variation in radius lengths")
        with col2:
            compactness_mean = st.number_input("Compactness Mean", 0.0, 0.4, 0.104, format="%.4f", help="Mean perimeter^2 / area - 1.0")
            concavity_mean = st.number_input("Concavity Mean", 0.0, 0.5, 0.089, format="%.4f", help="Mean severity of concave portions of the contour")
            concave_points_mean = st.number_input("Concave Points Mean", 0.0, 0.2, 0.048, format="%.4f", help="Mean number of concave portions of the contour")
            symmetry_mean = st.number_input("Symmetry Mean", 0.0, 0.4, 0.181, format="%.4f", help="Mean nuclear symmetry")
            fractal_dimension_mean = st.number_input("Fractal Dimension Mean", 0.0, 0.1, 0.063, format="%.4f", help="Mean 'coastline approximation' - 1.0")

    with tab2:
        st.markdown("<p style='color:#94A3B8; font-size:0.95rem;'>Standard error (SE) variations of nuclear attributes:</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            radius_se = st.number_input("Radius SE", 0.0, 3.0, 0.405, format="%.4f")
            texture_se = st.number_input("Texture SE", 0.0, 5.0, 1.216, format="%.4f")
            perimeter_se = st.number_input("Perimeter SE", 0.0, 22.0, 2.866, format="%.4f")
            area_se = st.number_input("Area SE", 0.0, 550.0, 40.34, format="%.2f")
            smoothness_se = st.number_input("Smoothness SE", 0.0, 0.03, 0.007, format="%.4f")
        with col2:
            compactness_se = st.number_input("Compactness SE", 0.0, 0.14, 0.025, format="%.4f")
            concavity_se = st.number_input("Concavity SE", 0.0, 0.4, 0.032, format="%.4f")
            concave_points_se = st.number_input("Concave Points SE", 0.0, 0.06, 0.012, format="%.4f")
            symmetry_se = st.number_input("Symmetry SE", 0.0, 0.08, 0.020, format="%.4f")
            fractal_dimension_se = st.number_input("Fractal Dimension SE", 0.0, 0.03, 0.004, format="%.4f")

    with tab3:
        st.markdown("<p style='color:#94A3B8; font-size:0.95rem;'>Worst/largest value (mean of the three largest values) for nuclear attributes:</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            radius_worst = st.number_input("Radius Worst", 0.0, 40.0, 16.27, format="%.2f")
            texture_worst = st.number_input("Texture Worst", 0.0, 50.0, 25.68, format="%.2f")
            perimeter_worst = st.number_input("Perimeter Worst", 0.0, 260.0, 107.26, format="%.2f")
            area_worst = st.number_input("Area Worst", 0.0, 4300.0, 880.58, format="%.2f")
            smoothness_worst = st.number_input("Smoothness Worst", 0.0, 0.23, 0.132, format="%.4f")
        with col2:
            compactness_worst = st.number_input("Compactness Worst", 0.0, 1.1, 0.254, format="%.4f")
            concavity_worst = st.number_input("Concavity Worst", 0.0, 1.3, 0.272, format="%.4f")
            concave_points_worst = st.number_input("Concave Points Worst", 0.0, 0.3, 0.115, format="%.4f")
            symmetry_worst = st.number_input("Symmetry Worst", 0.0, 0.7, 0.290, format="%.4f")
            fractal_dimension_worst = st.number_input("Fractal Dimension Worst", 0.0, 0.21, 0.084, format="%.4f")

    # Predict Button
    if st.button("🔍 Analyze Nuclear Features", use_container_width=True):
        input_data = np.array([[
            radius_mean, texture_mean, perimeter_mean, area_mean,
            smoothness_mean, compactness_mean, concavity_mean,
            concave_points_mean, symmetry_mean, fractal_dimension_mean,
            radius_se, texture_se, perimeter_se, area_se,
            smoothness_se, compactness_se, concavity_se,
            concave_points_se, symmetry_se, fractal_dimension_se,
            radius_worst, texture_worst, perimeter_worst, area_worst,
            smoothness_worst, compactness_worst, concavity_worst,
            concave_points_worst, symmetry_worst, fractal_dimension_worst
        ]])

        # Scale and Predict
        input_scaled = scalers['cancer'].transform(input_data)
        prediction = models['cancer'].predict(input_scaled)[0]
        probability = models['cancer'].predict_proba(input_scaled)[0]

        # Calculate Confidence Percentage
        prob_pct = (probability[1] if prediction == 1 else probability[0]) * 100

        # Output Styled Diagnostic Card
        if prediction == 1:
            result_html = f"""
            <div class="result-card result-card-positive">
                <div class="result-title">⚠️ High Risk: Malignant Detected</div>
                <div class="result-metric">The AI model classifies the cell nucleus features as <strong>Malignant</strong> with <strong>{prob_pct:.2f}%</strong> confidence.</div>
                <div class="progress-container">
                    <div class="progress-bar-fill" style="width: {prob_pct}%;"></div>
                </div>
                <div class="result-tips">
                    <strong>📋 Recommended Clinical Next Steps:</strong><br>
                    • Refer the patient to an oncologist or specialist for a diagnostic core biopsy.<br>
                    • Correlate findings with mammogram, ultrasound, or breast MRI reports.<br>
                    • Do not rely solely on automated screening; clinical correlation is necessary.
                </div>
            </div>
            """
        else:
            result_html = f"""
            <div class="result-card result-card-negative">
                <div class="result-title">✅ Low Risk: Benign Indicated</div>
                <div class="result-metric">The AI model classifies the cell nucleus features as <strong>Benign</strong> with <strong>{prob_pct:.2f}%</strong> confidence.</div>
                <div class="progress-container">
                    <div class="progress-bar-fill" style="width: {prob_pct}%;"></div>
                </div>
                <div class="result-tips">
                    <strong>📋 Patient Wellness & Screenings:</strong><br>
                    • Patient is advised to continue routine breast screenings according to standard clinical age groups.<br>
                    • Encourage standard self-checks and lifestyle wellness habits.
                </div>
            </div>
            """
        st.markdown(result_html, unsafe_allow_html=True)

# ================================================
# ===== DIABETES PREDICTION =====
# ================================================
elif disease == "Diabetes":
    st.markdown("### 🩺 Diabetes Prediction")
    st.markdown("Enter patient physiological indicators categorized below:")

    tab1, tab2 = st.tabs([
        "📊 Vital Measurements", 
        "🧬 Metabolic & Demographics"
    ])

    with tab1:
        st.markdown("<p style='color:#94A3B8; font-size:0.95rem;'>Key physiological markers:</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 3, help="Number of times pregnant (0 for males/children)")
            glucose = st.number_input("Glucose Level", 0, 200, 117, help="2-hour plasma glucose concentration in oral glucose tolerance test (mg/dL)")
        with col2:
            blood_pressure = st.number_input("Blood Pressure (Diastolic)", 0, 130, 72, help="Diastolic blood pressure (mm Hg)")
            skin_thickness = st.number_input("Skin Thickness (Triceps)", 0, 100, 23, help="Triceps skin fold thickness (mm)")

    with tab2:
        st.markdown("<p style='color:#94A3B8; font-size:0.95rem;'>Demographic profile and genetic markers:</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            insulin = st.number_input("Insulin Level", 0, 900, 30, help="2-hour serum insulin (mu U/ml)")
            bmi = st.number_input("BMI (Body Mass Index)", 0.0, 70.0, 32.0, help="Body mass index (weight in kg/(height in m)^2)")
        with col2:
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.47, help="Diabetes pedigree function (scores genetic history risk)")
            age = st.number_input("Age (Years)", 1, 100, 33)

    if st.button("🔍 Run Diabetes Assessment", use_container_width=True):
        input_data = np.array([[
            pregnancies, glucose, blood_pressure,
            skin_thickness, insulin, bmi, dpf, age
        ]])

        # Scale and Predict
        input_scaled = scalers['diabetes'].transform(input_data)
        prediction = models['diabetes'].predict(input_scaled)[0]
        probability = models['diabetes'].predict_proba(input_scaled)[0]

        # Calculate Confidence Percentage
        prob_pct = (probability[1] if prediction == 1 else probability[0]) * 100

        # Output Styled Diagnostic Card
        if prediction == 1:
            result_html = f"""
            <div class="result-card result-card-positive">
                <div class="result-title">⚠️ Diabetic Risk Detected</div>
                <div class="result-metric">The AI model predicts the patient is <strong>Diabetic</strong> with <strong>{prob_pct:.2f}%</strong> confidence.</div>
                <div class="progress-container">
                    <div class="progress-bar-fill" style="width: {prob_pct}%;"></div>
                </div>
                <div class="result-tips">
                    <strong>📋 Recommended Clinical Recommendations:</strong><br>
                    • Advise patient to consult an endocrinologist for a fasting blood sugar test and HbA1c screening.<br>
                    • Recommend dietary modifications: limit processed carbs, track daily glycemic index, and hydrate well.<br>
                    • Propose light-to-moderate physical activity (brisk walking) to aid insulin sensitivity.
                </div>
            </div>
            """
        else:
            result_html = f"""
            <div class="result-card result-card-negative">
                <div class="result-title">✅ Healthy Range Indicated</div>
                <div class="result-metric">The AI model predicts the patient is <strong>Not Diabetic</strong> with <strong>{prob_pct:.2f}%</strong> confidence.</div>
                <div class="progress-container">
                    <div class="progress-bar-fill" style="width: {prob_pct}%;"></div>
                </div>
                <div class="result-tips">
                    <strong>📋 Preventive Guidelines:</strong><br>
                    • Maintain a balanced diet rich in fibers, whole grains, and lean proteins.<br>
                    • Participate in 150 minutes of moderate aerobic exercise weekly.<br>
                    • Get annual physical screenings to monitor baseline glucose.
                </div>
            </div>
            """
        st.markdown(result_html, unsafe_allow_html=True)

# ================================================
# ===== HEART DISEASE PREDICTION =====
# ================================================
elif disease == "Heart Disease":
    st.markdown("### ❤️ Heart Disease Prediction")
    st.markdown("Enter patient medical history and diagnostic lab results:")

    tab1, tab2 = st.tabs([
        "👤 Patient Profile & History", 
        "🩺 Clinical Metrics & ECG"
    ])

    with tab1:
        st.markdown("<p style='color:#94A3B8; font-size:0.95rem;'>Demographics, history, and subjective indicators:</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (Years)", 1, 100, 54)
            sex = st.selectbox("Sex", ["Male", "Female"])
            chest_pain = st.selectbox("Chest Pain Type",
                        ["ATA", "NAP", "ASY", "TA"],
                        help="ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic chest pain, TA: Typical Angina")
        with col2:
            fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
            exercise_angina = st.selectbox("Exercise Angina (Does exercise trigger chest pain?)", ["No", "Yes"])

    with tab2:
        st.markdown("<p style='color:#94A3B8; font-size:0.95rem;'>Clinical tests, ECG, and stress exam results:</p>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            resting_bp = st.number_input("Resting Blood Pressure", 0, 250, 130, help="Resting blood pressure (mm Hg)")
            cholesterol = st.number_input("Cholesterol", 0, 700, 223, help="Serum cholesterol (mg/dl)")
            resting_ecg = st.selectbox("Resting ECG Results",
                         ["Normal", "ST", "LVH"],
                         help="Normal, ST: ST-T wave abnormalities, LVH: Left ventricular hypertrophy by Estes' criteria")
        with col2:
            max_hr = st.number_input("Max Heart Rate Achieved", 0, 250, 136, help="Peak heart rate during exercise test")
            oldpeak = st.number_input("Oldpeak (ST Depression)", -3.0, 7.0, 0.89, help="ST depression induced by exercise relative to rest")
            st_slope = st.selectbox("ST Slope Type",
                       ["Up", "Flat", "Down"],
                       help="The slope of the peak exercise ST segment")

    if st.button("🔍 Assess Cardiovascular Risk", use_container_width=True):
        # Encode inputs
        sex_val = 1 if sex == "Male" else 0
        fasting_bs_val = 1 if fasting_bs == "Yes" else 0
        exercise_angina_val = 1 if exercise_angina == "Yes" else 0

        # One hot encode
        cp_ata = 1 if chest_pain == "ATA" else 0
        cp_nap = 1 if chest_pain == "NAP" else 0
        cp_asy = 1 if chest_pain == "ASY" else 0
        cp_ta = 1 if chest_pain == "TA" else 0

        ecg_lvh = 1 if resting_ecg == "LVH" else 0
        ecg_normal = 1 if resting_ecg == "Normal" else 0
        ecg_st = 1 if resting_ecg == "ST" else 0

        slope_down = 1 if st_slope == "Down" else 0
        slope_flat = 1 if st_slope == "Flat" else 0
        slope_up = 1 if st_slope == "Up" else 0

        input_data = np.array([[
            age, sex_val, resting_bp, cholesterol,
            fasting_bs_val, max_hr, exercise_angina_val, oldpeak,
            cp_asy, cp_ata, cp_nap, cp_ta,
            ecg_lvh, ecg_normal, ecg_st,
            slope_down, slope_flat, slope_up
        ]])

        # Scale and Predict
        input_scaled = scalers['heart'].transform(input_data)
        prediction = models['heart'].predict(input_scaled)[0]
        probability = models['heart'].predict_proba(input_scaled)[0]

        # Calculate Confidence Percentage
        prob_pct = (probability[1] if prediction == 1 else probability[0]) * 100

        # Output Styled Diagnostic Card
        if prediction == 1:
            result_html = f"""
            <div class="result-card result-card-positive">
                <div class="result-title">⚠️ Cardiovascular Risk Detected</div>
                <div class="result-metric">The AI model predicts <strong>Heart Disease</strong> is present with <strong>{prob_pct:.2f}%</strong> confidence.</div>
                <div class="progress-container">
                    <div class="progress-bar-fill" style="width: {prob_pct}%;"></div>
                </div>
                <div class="result-tips">
                    <strong>📋 Recommended Clinical Next Steps:</strong><br>
                    • Refer the patient to a cardiologist for diagnostic verification (ECG, echocardiogram, coronary angiogram).<br>
                    • Review current cholesterol/lipid panel, diet, and lifestyle.<br>
                    • Recommend blood pressure monitoring and immediate cessation of smoking/tobacco if applicable.
                </div>
            </div>
            """
        else:
            result_html = f"""
            <div class="result-card result-card-negative">
                <div class="result-title">✅ Low Cardiovascular Risk</div>
                <div class="result-metric">The AI model predicts <strong>No Heart Disease</strong> with <strong>{prob_pct:.2f}%</strong> confidence.</div>
                <div class="progress-container">
                    <div class="progress-bar-fill" style="width: {prob_pct}%;"></div>
                </div>
                <div class="result-tips">
                    <strong>📋 Heart Healthy Habits:</strong><br>
                    • Continue cardiorespiratory exercises (30 mins of moderate activity 5 days/week).<br>
                    • Maintain a diet rich in omega-3, healthy fats, fiber, and low sodium.<br>
                    • Track cholesterol levels, blood pressure, and blood glucose in routine annual checks.
                </div>
            </div>
            """
        st.markdown(result_html, unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown(
    """
    <div class="footer-container">
        <p>⚠️ <b>Medical Disclaimer:</b> DiagnoAI is a screening tool designed for educational and supportive clinical inquiry. It does not replace professional medical advice, diagnosis, or clinical evaluation by a physician.</p>
        <p style="margin-top: 5px; opacity: 0.6;">&copy; 2026 DiagnoAI — Multi Disease Prediction System. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)