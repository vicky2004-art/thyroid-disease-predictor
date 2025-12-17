import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import os

# --- Page Configuration ---
st.set_page_config(page_title="Thyroid Disease Prediction", page_icon="🏥", layout="wide")

# --- Robust Data Loader ---
@st.cache_resource
def train_model():
    # 1. Check if file exists
    if not os.path.exists('Thyroid-Dataset.csv'):
        return None, None, None, "File 'Thyroid-Dataset.csv' not found. Please place it in the same folder as this script.", None

    try:
        df = pd.read_csv('Thyroid-Dataset.csv')
        
        # 2. Normalize Column Names
        df.columns = [c.lower().strip() for c in df.columns]

        # 3. Clean 'Sex' (Map F/M to 0/1)
        if 'sex' in df.columns:
            df['sex'] = df['sex'].map({'F': 0, 'M': 1, 'f': 0, 'm': 1})
            df['sex'] = df['sex'].fillna(0) # Default to 0

        # 4. Clean Boolean Columns (True/False -> 1/0)
        # We explicitly target columns that are likely boolean
        for col in df.columns:
            if col not in ['age', 'tsh', 't3', 'tt4', 't4u', 'fti', 'class', 'referral source']:
                # Convert to string, lower case, then map
                df[col] = df[col].astype(str).str.lower().map({
                    't': 1, 'f': 0, 'true': 1, 'false': 0, '1': 1, '0': 0
                }).fillna(0)

        # 5. Clean Numeric Columns (Handle '?' and text)
        numeric_cols = ['age', 'tsh', 't3', 'tt4', 't4u', 'fti']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') # Invalid turns to NaN

        # 6. Impute Missing Values
        # Drop referral source if exists
        if 'referral source' in df.columns:
            df = df.drop('referral source', axis=1)
            
        # Fill NaN with median
        imputer = SimpleImputer(strategy='median')
        feature_cols = [c for c in df.columns if c != 'class']
        df[feature_cols] = imputer.fit_transform(df[feature_cols])

        # 7. Train KNN
        le = LabelEncoder()
        y = le.fit_transform(df['class'])
        X = df[feature_cols]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_scaled, y)
        
        return knn, scaler, le, None, feature_cols

    except Exception as e:
        return None, None, None, str(e), None

# --- App Logic ---
st.title("🏥 Thyroid Disease Prediction System")

knn, scaler, le, error_msg, feature_cols = train_model()

if error_msg:
    st.error(f"⚠️ System Error: {error_msg}")
    st.stop()

# --- User Interface ---
with st.form("patient_data"):
    st.header("Patient Data Entry")
    
    col1, col2, col3 = st.columns(3)
    name = col1.text_input("Patient Name")
    age = col2.number_input("Age", 1, 120, 30)
    gender_txt = col3.selectbox("Gender", ["Female", "Male"])
    sex = 0 if gender_txt == "Female" else 1

    st.subheader("Clinical History")
    c1, c2, c3 = st.columns(3)
    with c1:
        on_thyroxine = st.checkbox("On Thyroxine")
        on_antithyroid = st.checkbox("On Antithyroid Meds")
        sick = st.checkbox("Sick / Ill")
        pregnant = st.checkbox("Pregnant")
    with c2:
        surgery = st.checkbox("Thyroid Surgery")
        i131 = st.checkbox("I131 Treatment")
        lithium = st.checkbox("On Lithium")
        goitre = st.checkbox("Goitre")
    with c3:
        tumor = st.checkbox("Tumor")
        hypopituitary = st.checkbox("Hypopituitary")
        psych = st.checkbox("Psychological Symptoms")
    
    st.subheader("Lab Results")
    l1, l2, l3, l4, l5 = st.columns(5)
    tsh = l1.number_input("TSH", value=1.5)
    t3 = l2.number_input("T3", value=2.0)
    tt4 = l3.number_input("TT4", value=100.0)
    t4u = l4.number_input("T4U", value=1.0)
    fti = l5.number_input("FTI", value=100.0)

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    # Map inputs to the exact feature columns found in CSV
    input_data = {col: 0 for col in feature_cols} # Init with 0
    
    # Fill Data
    input_data['age'] = age
    input_data['sex'] = sex
    input_data['tsh'] = tsh
    input_data['t3'] = t3
    input_data['tt4'] = tt4
    input_data['t4u'] = t4u
    input_data['fti'] = fti
    
    # Map booleans safely
    bool_map = {
        'on thyroxine': on_thyroxine, 'on antithyroid medication': on_antithyroid,
        'sick': sick, 'pregnant': pregnant, 'thyroid surgery': surgery,
        'i131 treatment': i131, 'lithium': lithium, 'goitre': goitre,
        'tumor': tumor, 'hypopituitary': hypopituitary, 'psych': psych
    }
    
    for key, val in bool_map.items():
        if key in input_data:
            input_data[key] = int(val)

    # Predict
    vector = [input_data[c] for c in feature_cols]
    vector_scaled = scaler.transform([vector])
    pred_idx = knn.predict(vector_scaled)[0]
    result = le.inverse_transform([pred_idx])[0]

    st.divider()
    st.subheader(f"Results for {name}")
    
    if result == 'negative':
        st.success(f"**Diagnosis: {result.upper()}** (Healthy)")
        st.write("No thyroid disease detected.")
    else:
        st.warning(f"**Diagnosis: {result.upper()}**")
        st.write("Thyroid abnormality detected. Please consult a specialist.")