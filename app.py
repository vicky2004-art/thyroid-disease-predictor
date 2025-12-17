import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Page Configuration ---
st.set_page_config(
    page_title="Thyroid Health Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS for User Friendly Look ---
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #4F8BF9; text-align: center; margin-bottom: 20px;}
    .sub-header {font-size: 1.5rem; color: #444; margin-top: 20px;}
    .report-box {background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4F8BF9;}
    .stButton>button {width: 100%; background-color: #4F8BF9; color: white; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- 1. Data Loading & Training Engine ---
@st.cache_resource
def build_model():
    try:
        # Load Data
        df = pd.read_csv('Thyroid-Dataset.csv')
        
        # preprocessing: Drop ID-like columns
        if 'referral source' in df.columns:
            df = df.drop(['referral source'], axis=1)

        # 1. Clean Sex Column
        df['sex'] = df['sex'].map({'F': 0, 'M': 1})
        df['sex'] = df['sex'].fillna(0)  # Default to Female if unknown

        # 2. Clean Boolean Columns (ensure they are 0/1 integers)
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df[col] = df[col].astype(int)

        # 3. Handle Numeric Missing Values (Impute with Median)
        # Identify numeric columns (excluding the target 'class')
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        # 4. Target Encoding
        le = LabelEncoder()
        df['class'] = le.fit_transform(df['class'])

        # Split Data
        X = df.drop('class', axis=1)
        y = df['class']

        # Scale Data (Important for KNN)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate Accuracy for Transparency
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, knn.predict(X_test))
        
        # Refit on full data for the app
        knn.fit(X_scaled, y)

        return knn, scaler, le, X.columns, acc

    except Exception as e:
        return None, None, None, str(e), 0

# Load the system
knn, scaler, le, feature_names, accuracy = build_model()

# --- 2. Sidebar Input Section ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("Patient Data Input")
    st.write("Enter patient details below.")
    
    st.subheader("1. Demographics")
    name = st.text_input("Patient Name", "Guest")
    age = st.slider("Age", 1, 100, 30)
    sex_label = st.radio("Gender", ["Female", "Male"], horizontal=True)
    sex = 0 if sex_label == "Female" else 1

    st.subheader("2. Lab Results")
    st.info("Leave default if unknown (uses population average).")
    tsh = st.number_input("TSH (Thyroid Stimulating Hormone)", value=1.6)
    t3 = st.number_input("T3 Level", value=2.0)
    tt4 = st.number_input("TT4 Level", value=100.0)
    t4u = st.number_input("T4U Level", value=0.98)
    fti = st.number_input("FTI Level", value=107.0)

# --- 3. Main Dashboard ---
st.markdown('<div class="main-header">🏥 Thyroid Disease Prediction System</div>', unsafe_allow_html=True)

if knn is None:
    st.error(f"⚠️ Critical Error: {feature_names}")
    st.warning("Ensure 'Thyroid-Dataset.csv' is in the exact same folder as this script.")
    st.stop()

# Display System Status
col1, col2 = st.columns(2)
col1.metric("Model Algorithm", "K-Nearest Neighbors")
col2.metric("Model Precision", f"{accuracy:.1%}")

st.divider()

# Symptoms Form (Grouped for better UX)
st.subheader("3. Clinical Symptoms & History")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**General Condition**")
    sick = st.checkbox("Sick / Ill")
    pregnant = st.checkbox("Pregnant")
    tumor = st.checkbox("Tumor")
    goitre = st.checkbox("Goitre (Neck Swelling)")

with col_b:
    st.markdown("**Medication History**")
    on_thyroxine = st.checkbox("On Thyroxine")
    on_antithyroid = st.checkbox("On Antithyroid Meds")
    lithium = st.checkbox("On Lithium")
    i131_treatment = st.checkbox("I131 Treatment History")

with col_c:
    st.markdown("**Clinical Queries**")
    query_hypo = st.checkbox("Query: Hypothyroid")
    query_hyper = st.checkbox("Query: Hyperthyroid")
    psych = st.checkbox("Psychological Symptoms")
    # Hidden defaults for less common fields to keep UI clean
    query_on_thyroxine = False
    thyroid_surgery = False
    hypopituitary = False

# --- 4. Prediction Logic ---
if st.button("Analyze & Predict Disease"):
    
    # Map inputs to features
    # NOTE: Order must match training data: 
    # age, sex, on thyroxine, query on thyroxine, on antithyroid medication, sick, pregnant, thyroid surgery, 
    # I131 treatment, query hypothyroid, query hyperthyroid, lithium, goitre, tumor, hypopituitary, psych, 
    # TSH, T3, TT4, T4U, FTI
    
    input_vector = [
        age, sex,
        int(on_thyroxine), int(query_on_thyroxine), int(on_antithyroid),
        int(sick), int(pregnant), int(thyroid_surgery),
        int(i131_treatment), int(query_hypo), int(query_hyper),
        int(lithium), int(goitre), int(tumor),
        int(hypopituitary), int(psych),
        tsh, t3, tt4, t4u, fti
    ]
    
    # Scale and Predict
    input_scaled = scaler.transform([input_vector])
    prediction_idx = knn.predict(input_scaled)[0]
    prediction_label = le.inverse_transform([prediction_idx])[0]
    
    # Display Result
    st.markdown("---")
    st.subheader(f"📋 Medical Report for: {name}")
    
    # Dynamic Styling for Result
    if prediction_label == 'negative':
        st.success(f"### Diagnosis: {prediction_label.upper()} (Normal)")
        st.write("No significant thyroid abnormalities detected based on the provided parameters.")
    else:
        st.error(f"### Diagnosis: {prediction_label.upper()}")
        st.write("Potential thyroid disorder detected.")
    
    # Detailed Solutions Dictionary
    solutions = {
        'negative': ["Maintain a balanced diet rich in iodine.", "Regular annual checkups recommended."],
        'hypothyroid conditions': ["Consult an Endocrinologist immediately.", "Possible treatment: Levothyroxine replacement therapy.", "Diet: Increase selenium and zinc intake."],
        'hyperthyroid conditions': ["Consult an Endocrinologist immediately.", "Possible treatment: Antithyroid medications (methimazole) or radioactive iodine.", "Diet: Avoid excessive iodine."],
        'binding protein': ["This indicates abnormal protein levels binding to thyroid hormones.", "Further specific blood tests required."],
        'general health': ["Condition may be related to non-thyroid general health factors.", "Consult a General Physician."],
        'replacement therapy': ["Patient is currently under hormone replacement.", "Monitor TSH levels regularly to adjust dosage."],
        'miscellaneous': ["Complex clinical presentation.", "Detailed full-body checkup suggested."],
        'discordant results': ["Lab values contradict each other.", "Redo TSH and T4 tests to confirm."],
        'antithyroid treatment': ["Patient is undergoing treatment.", "Monitor for side effects of medication."]
    }
    
    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown("**Recommended Actions:**")
    for step in solutions.get(prediction_label, ["Consult a specialist for a detailed diagnosis."]):
        st.markdown(f"- {step}")
    st.markdown('</div>', unsafe_allow_html=True)