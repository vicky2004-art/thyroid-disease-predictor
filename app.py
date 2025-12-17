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
    layout="wide"
)

# --- Robust Data Loading Function ---
@st.cache_resource
def load_and_train():
    try:
        # 1. Load Data
        df = pd.read_csv('Thyroid-Dataset.csv')
        
        # 2. Normalize Column Names (lowercase, strip spaces)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # 3. Drop ID columns
        if 'referral source' in df.columns:
            df = df.drop(['referral source'], axis=1)

        # 4. Force Numeric Conversion (Crucial for avoiding '?' errors)
        # We explicitly list potential numeric columns
        numeric_candidates = ['age', 'tsh', 't3', 'tt4', 't4u', 'fti']
        for col in numeric_candidates:
            if col in df.columns:
                # errors='coerce' turns '?' and text into NaN (numbers)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 5. Clean 'Sex' Column
        if 'sex' in df.columns:
            df['sex'] = df['sex'].map({'F': 0, 'M': 1, 'f': 0, 'm': 1})
            df['sex'] = df['sex'].fillna(0) # Default to Female

        # 6. Clean Boolean Columns
        # Detect columns that look like booleans or objects that should be booleans
        for col in df.columns:
            if col not in numeric_candidates and col != 'class' and col != 'sex':
                # Map True/False text to 1/0
                df[col] = df[col].astype(str).str.lower().map({
                    't': 1, 'f': 0, 'true': 1, 'false': 0, '1': 1, '0': 0
                })
                df[col] = df[col].fillna(0)

        # 7. Impute Missing Values (Fill NaNs with Median)
        imputer = SimpleImputer(strategy='median')
        # Select all columns except 'class'
        feature_cols = [c for c in df.columns if c != 'class']
        df[feature_cols] = imputer.fit_transform(df[feature_cols])

        # 8. Encode Target
        le = LabelEncoder()
        df['class'] = le.fit_transform(df['class'])
        
        # 9. Train Model
        X = df[feature_cols]
        y = df['class']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_scaled, y)
        
        # Calculate accuracy for display
        score = knn.score(X_scaled, y)
        
        return knn, scaler, le, feature_cols, score

    except Exception as e:
        return None, None, None, str(e), 0

# Initialize App
knn, scaler, le, feature_cols, accuracy = load_and_train()

# --- User Interface ---
st.title("🩺 Thyroid Disease Prediction System")
st.markdown(f"**System Status:** {'🟢 Online' if knn else '🔴 Offline'}")

if knn is None:
    st.error(f"Error loading data: {feature_cols}")
    st.info("Please ensure 'Thyroid-Dataset.csv' is in the same folder and is not corrupted.")
    st.stop()

st.sidebar.header("Patient Vitals")
with st.form("entry_form"):
    st.subheader("1. Patient Details")
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Name")
    age = c2.number_input("Age", 1, 120, 30)
    sex_val = c3.selectbox("Gender", ["Female", "Male"])
    sex = 0 if sex_val == "Female" else 1

    st.subheader("2. Lab Report Values")
    st.caption("If a value is unknown, leave it as the default average.")
    l1, l2, l3, l4, l5 = st.columns(5)
    tsh = l1.number_input("TSH", value=1.5)
    t3 = l2.number_input("T3", value=2.0)
    tt4 = l3.number_input("TT4", value=100.0)
    t4u = l4.number_input("T4U", value=1.0)
    fti = l5.number_input("FTI", value=100.0)

    st.subheader("3. Clinical History")
    h1, h2, h3 = st.columns(3)
    with h1:
        on_thyroxine = st.checkbox("On Thyroxine")
        on_antithyroid = st.checkbox("On Antithyroid Meds")
        sick = st.checkbox("Sick / Ill")
        pregnant = st.checkbox("Pregnant")
    with h2:
        thyroid_surgery = st.checkbox("Thyroid Surgery History")
        i131 = st.checkbox("I131 Treatment")
        lithium = st.checkbox("On Lithium")
        goitre = st.checkbox("Goitre Present")
    with h3:
        tumor = st.checkbox("Tumor Present")
        hypopituitary = st.checkbox("Hypopituitary")
        psych = st.checkbox("Psychological Symptoms")
        
    submit = st.form_submit_button("Generate Prediction")

if submit:
    # Construct input array matching the TRAINING columns exactly
    # We create a dictionary first to map values to column names
    input_dict = {col: 0 for col in feature_cols} # Initialize all with 0
    
    # Fill known values
    input_dict['age'] = age
    input_dict['sex'] = sex
    input_dict['tsh'] = tsh
    input_dict['t3'] = t3
    input_dict['tt4'] = tt4
    input_dict['t4u'] = t4u
    input_dict['fti'] = fti
    
    # Fill booleans
    # Note: Keys must match the lowercased column names from CSV
    input_dict['on thyroxine'] = int(on_thyroxine)
    input_dict['on antithyroid medication'] = int(on_antithyroid)
    input_dict['sick'] = int(sick)
    input_dict['pregnant'] = int(pregnant)
    input_dict['thyroid surgery'] = int(thyroid_surgery)
    input_dict['i131 treatment'] = int(i131)
    input_dict['lithium'] = int(lithium)
    input_dict['goitre'] = int(goitre)
    input_dict['tumor'] = int(tumor)
    input_dict['hypopituitary'] = int(hypopituitary)
    input_dict['psych'] = int(psych)
    
    # Convert dictionary to list in correct order
    input_vector = [input_dict[col] for col in feature_cols]
    
    # Predict
    input_scaled = scaler.transform([input_vector])
    pred_idx = knn.predict(input_scaled)[0]
    pred_label = le.inverse_transform([pred_idx])[0]
    
    # Output
    st.divider()
    st.subheader(f"Results for {name}")
    
    if pred_label == 'negative':
        st.success(f"**Diagnosis: {pred_label.upper()}** (Healthy)")
        st.write("No thyroid disease detected.")
    else:
        st.warning(f"**Diagnosis: {pred_label.upper()}**")
        st.write("Condition detected. Please see the advice below.")
        
    # Advice Logic
    advice = {
        'negative': "Maintain healthy diet and regular exercise.",
        'hypothyroid conditions': "Consult an Endocrinologist. TSH levels may be high. Treatment often involves hormone replacement (Levothyroxine).",
        'hyperthyroid conditions': "Consult an Endocrinologist. TSH levels may be low. Treatment may involve antithyroid drugs.",
        'binding protein': "Abnormal protein binding detected. This affects total hormone levels but free hormone levels might be normal. Specialist review needed.",
        'general health': "General health checkup recommended.",
        'discordant results': "Lab error or inconsistent data. Repeat blood tests.",
        'replacement therapy': "Patient is on therapy. Continue monitoring."
    }
    
    st.info(f"**Recommendation:** {advice.get(pred_label, 'Consult a doctor for detailed analysis.')}")