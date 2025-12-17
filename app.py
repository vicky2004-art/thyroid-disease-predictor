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
    page_title="Thyroid Health AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #0E1117;
        text-align: center;
        font-weight: 800;
        margin-bottom: 20px;
    }
    .result-box-negative {
        padding: 20px;
        background-color: #D4EDDA;
        color: #155724;
        border-radius: 10px;
        border-left: 5px solid #28A745;
        text-align: center;
    }
    .result-box-positive {
        padding: 20px;
        background-color: #F8D7DA;
        color: #721C24;
        border-radius: 10px;
        border-left: 5px solid #DC3545;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading & Model Training ---
@st.cache_resource
def load_and_train_model():
    try:
        # Load Data
        df = pd.read_csv('Thyroid-Dataset.csv')
        
        # 1. Clean Column Names
        df.columns = [c.lower().strip() for c in df.columns]
        
        # 2. Preprocessing
        # Drop irrelevant ID columns
        if 'referral source' in df.columns:
            df = df.drop(['referral source'], axis=1)
            
        # Clean 'Sex'
        if 'sex' in df.columns:
            df['sex'] = df['sex'].map({'F': 0, 'M': 1, 'f': 0, 'm': 1})
            df['sex'] = df['sex'].fillna(0) # Default to Female

        # Clean Boolean Columns
        # We identify columns that are boolean or object type (excluding class)
        for col in df.columns:
            if col != 'class' and col != 'sex' and col != 'age':
                # Try to convert to numeric first to catch floats
                df[col] = pd.to_numeric(df[col], errors='ignore')
                # If still object/bool, map standard true/false values
                if df[col].dtype == 'bool' or df[col].dtype == 'object':
                     df[col] = df[col].astype(str).str.lower().map({
                        't': 1, 'f': 0, 'true': 1, 'false': 0, '1': 1, '0': 0
                    }).fillna(0)

        # Handle Numeric Missing Values (Imputation)
        numeric_cols = ['age', 'tsh', 't3', 'tt4', 't4u', 'fti']
        for col in numeric_cols:
             if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        imputer = SimpleImputer(strategy='median')
        feature_cols = [c for c in df.columns if c != 'class']
        df[feature_cols] = imputer.fit_transform(df[feature_cols])

        # Encode Target
        le = LabelEncoder()
        df['class'] = le.fit_transform(df['class'])
        
        # Train Model
        X = df[feature_cols]
        y = df['class']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split for accuracy check
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, knn.predict(X_test))
        
        # Retrain on full data for best performance
        knn.fit(X_scaled, y)
        
        return knn, scaler, le, feature_cols, accuracy

    except Exception as e:
        return None, None, None, str(e), 0

# Initialize System
knn, scaler, le, feature_cols, model_acc = load_and_train_model()

# --- Main Interface ---
st.markdown('<div class="main-header">🏥 Thyroid Disease Prediction System</div>', unsafe_allow_html=True)

if knn is None:
    st.error(f"System Error: {feature_cols}")
    st.warning("Please ensure 'Thyroid-Dataset.csv' is in the same folder.")
    st.stop()

# Sidebar Information
with st.sidebar:
    st.title("System Info")
    st.info(f"**Model Accuracy:** {model_acc:.1%}")
    st.write("This system uses the K-Nearest Neighbors (KNN) algorithm to analyze patient symptoms and lab data.")
    st.markdown("---")
    st.write("**Created by:** Thyroid AI Team")

# Input Form
with st.form("patient_form"):
    st.subheader("1. Patient Profile")
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Patient Name")
    age = c2.number_input("Age", min_value=1, max_value=120, value=30)
    sex_input = c3.selectbox("Gender", ["Female", "Male"])
    sex = 0 if sex_input == "Female" else 1

    st.subheader("2. Symptoms & History")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("**General Status**")
        sick = st.checkbox("Sick / Ill")
        pregnant = st.checkbox("Pregnant")
        tumor = st.checkbox("Tumor detected")
        goitre = st.checkbox("Goitre (Neck Swelling)")
    
    with col_b:
        st.markdown("**Medications**")
        on_thyroxine = st.checkbox("On Thyroxine")
        on_antithyroid = st.checkbox("On Antithyroid Meds")
        lithium = st.checkbox("On Lithium")
        i131 = st.checkbox("I131 Treatment History")

    with col_c:
        st.markdown("**Clinical Observations**")
        surgery = st.checkbox("Thyroid Surgery History")
        hypopituitary = st.checkbox("Hypopituitary")
        psych = st.checkbox("Psychological Symptoms")
        # Hidden inputs for less common queries to keep UI clean (defaulting to False)
        query_hypo = False
        query_hyper = False
        query_thyroxine = False

    st.subheader("3. Lab Reports (Leave default if unknown)")
    l1, l2, l3, l4, l5 = st.columns(5)
    tsh = l1.number_input("TSH", value=1.5, help="Thyroid Stimulating Hormone")
    t3 = l2.number_input("T3", value=2.0, help="Triiodothyronine")
    tt4 = l3.number_input("TT4", value=100.0, help="Total Thyroxine")
    t4u = l4.number_input("T4U", value=1.0, help="Thyroxine Utilization")
    fti = l5.number_input("FTI", value=100.0, help="Free Thyroxine Index")

    submit_btn = st.form_submit_button("Analyze & Predict")

# --- Prediction Logic ---
if submit_btn:
    if not name:
        st.warning("Please enter the patient's name.")
    else:
        # Create input dictionary with default 0s
        input_data = {col: 0 for col in feature_cols}
        
        # Fill standard values
        input_data['age'] = age
        input_data['sex'] = sex
        input_data['tsh'] = tsh
        input_data['t3'] = t3
        input_data['tt4'] = tt4
        input_data['t4u'] = t4u
        input_data['fti'] = fti
        
        # Fill booleans
        # Map checkbox variables to the column names
        bool_map = {
            'on thyroxine': on_thyroxine, 'on antithyroid medication': on_antithyroid,
            'sick': sick, 'pregnant': pregnant, 'thyroid surgery': surgery,
            'i131 treatment': i131, 'lithium': lithium, 'goitre': goitre,
            'tumor': tumor, 'hypopituitary': hypopituitary, 'psych': psych,
            'query hypothyroid': query_hypo, 'query hyperthyroid': query_hyper,
            'query on thyroxine': query_thyroxine
        }
        
        for k, v in bool_map.items():
            if k in input_data:
                input_data[k] = int(v)

        # Convert to vector matching training order
        vector = [input_data[c] for c in feature_cols]
        
        # Scale and Predict
        vector_scaled = scaler.transform([vector])
        pred_idx = knn.predict(vector_scaled)[0]
        result = le.inverse_transform([pred_idx])[0]
        
        # --- Display Results ---
        st.divider()
        st.subheader(f"📋 Analysis Report for {name}")
        
        if result == 'negative':
            st.markdown(f"""
                <div class="result-box-negative">
                    <h2>NEGATIVE (Normal)</h2>
                    <p>No significant thyroid disease detected based on the inputs.</p>
                </div>
            """, unsafe_allow_html=True)
            st.success("Recommendation: Maintain a healthy lifestyle and regular annual checkups.")
        else:
            condition_name = result.replace(" conditions", "").title()
            st.markdown(f"""
                <div class="result-box-positive">
                    <h2>POSITIVE: {condition_name}</h2>
                    <p>The analysis indicates a high probability of {condition_name}.</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Specific Advice Logic
            advice = {
                'hypothyroid': "Consider consulting an Endocrinologist for TSH/T4 testing. Treatment often includes hormone replacement.",
                'hyperthyroid': "Consult a specialist. Treatment may involve antithyroid medication or iodine therapy.",
                'binding protein': "This suggests an anomaly in protein levels, not necessarily the thyroid gland itself. Further testing needed.",
                'general health': "This classification suggests general health issues rather than a specific thyroid pathology.",
                'discordant results': "The lab data provided is inconsistent. It is highly recommended to repeat the blood tests."
            }
            
            # Find best advice match
            found_advice = "Consult a doctor for a detailed diagnosis and treatment plan."
            for key in advice:
                if key in result:
                    found_advice = advice[key]
            
            st.error(f"**Medical Advice:** {found_advice}")