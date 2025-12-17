import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Page Configuration
st.set_page_config(page_title="Thyroid Disease Prediction", layout="wide")

# --- 1. Load and Preprocess Data (Debugged) ---
@st.cache_resource
def load_and_train_model():
    try:
        # Load dataset
        df = pd.read_csv('Thyroid-Dataset.csv')
        
        # FIX: Force column names to lower case to avoid case-sensitivity errors
        df.columns = [x.lower() for x in df.columns]

        # Drop irrelevant columns
        if 'referral source' in df.columns:
            df = df.drop(['referral source'], axis=1)

        # FIX: Handle 'Age' column which might have '?' or be read as text
        # Force convert to numeric, turn errors (like '?') into NaN
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['age'] = df['age'].fillna(df['age'].median())

        # Clean 'sex' column
        if 'sex' in df.columns:
            df['sex'] = df['sex'].map({'F': 0, 'M': 1})
            df['sex'] = df['sex'].fillna(0)  # Default to 0 (Female) if unknown

        # FIX: Handle Boolean columns explicitly (sometimes read as strings "False"/"True")
        # We look for object columns that look like booleans
        for col in df.columns:
            if df[col].dtype == 'object' and col != 'class':
                # Try replacing False/True text with 0/1
                df[col] = df[col].replace({'f': 0, 'F': 0, 'False': 0, 't': 1, 'T': 1, 'True': 1})
        
        # Now convert all remaining boolean-like columns to integers
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df[col] = df[col].astype(int)

        # Handle Missing Values in Numeric Columns
        # We explicitly list the lab columns
        lab_cols = ['tsh', 't3', 'tt4', 't4u', 'fti']
        for col in lab_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Impute all numeric columns (Age + Labs)
        imputer = SimpleImputer(strategy='median')
        numeric_columns = ['age'] + [c for c in lab_cols if c in df.columns]
        df[numeric_columns] = imputer.fit_transform(df[numeric_columns])

        # Encode Target Variable
        le = LabelEncoder()
        df['class'] = le.fit_transform(df['class'])

        # Features and Target
        X = df.drop('class', axis=1)
        y = df['class']

        # Scale Features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train KNN Model
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_scaled, y)

        return knn, scaler, le, X.columns

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

# Load model components
knn, scaler, le, feature_cols = load_and_train_model()

# Solution Mapping
solutions_map = {
    'negative': "No thyroid disease detected. Maintain a healthy lifestyle.",
    'hypothyroid conditions': "Possible Hypothyroidism. Consult an endocrinologist.",
    'general health': "General health check recommended.",
    'binding protein': "Abnormal binding protein levels. Consult a specialist.",
    'replacement therapy': "Patient is on hormone replacement therapy. Monitor levels.",
    'discordant results': "Inconsistent lab results. Repeat tests recommended.",
    'hyperthyroid conditions': "Possible Hyperthyroidism. Consult an endocrinologist.",
    'miscellaneous': "Detailed clinical evaluation needed.",
    'antithyroid treatment': "Patient is on antithyroid treatment. Continue monitoring."
}

# --- 2. Website UI ---
st.title("🏥 Thyroid Disease Prediction System")

if knn is None:
    st.warning("⚠️ Data could not be loaded. Please ensure 'Thyroid-Dataset.csv' is in the same folder.")
    st.stop()

with st.form("patient_form"):
    st.header("Patient Details")
    col1, col2, col3 = st.columns(3)
    name = col1.text_input("Patient Name")
    age = col2.number_input("Age", min_value=1, max_value=120, value=30)
    sex_input = col3.selectbox("Gender", ["Female", "Male"])
    sex = 0 if sex_input == "Female" else 1

    st.header("Symptoms & History")
    col_sym1, col_sym2, col_sym3 = st.columns(3)
    
    # We map these manual inputs to the exact column names expected by the model
    # Note: We must match the order or name used in training.
    
    with col_sym1:
        sick = st.checkbox("Sick / Ill")
        pregnant = st.checkbox("Pregnant")
        thyroid_surgery = st.checkbox("Thyroid Surgery")
        goitre = st.checkbox("Goitre")
        tumor = st.checkbox("Tumor")
        
    with col_sym2:
        on_thyroxine = st.checkbox("On Thyroxine")
        query_on_thyroxine = st.checkbox("Query On Thyroxine")
        on_antithyroid = st.checkbox("On Antithyroid Meds")
        i131_treatment = st.checkbox("I131 Treatment")
        
    with col_sym3:
        query_hypo = st.checkbox("Query Hypothyroid")
        query_hyper = st.checkbox("Query Hyperthyroid")
        lithium = st.checkbox("On Lithium")
        hypopituitary = st.checkbox("Hypopituitary")
        psych = st.checkbox("Psychological Issues")

    st.header("Lab Results")
    c1, c2, c3, c4, c5 = st.columns(5)
    tsh = c1.number_input("TSH", value=1.4)
    t3 = c2.number_input("T3", value=2.0)
    tt4 = c3.number_input("TT4", value=100.0)
    t4u = c4.number_input("T4U", value=0.98)
    fti = c5.number_input("FTI", value=100.0)

    submit = st.form_submit_button("Predict")

if submit:
    # Construct a DataFrame with the exact same columns as training data
    # This is safer than a list because order doesn't matter as much
    input_data = {
        'age': [age],
        'sex': [sex],
        'on thyroxine': [int(on_thyroxine)],
        'query on thyroxine': [int(query_on_thyroxine)],
        'on antithyroid medication': [int(on_antithyroid)],
        'sick': [int(sick)],
        'pregnant': [int(pregnant)],
        'thyroid surgery': [int(thyroid_surgery)],
        'i131 treatment': [int(i131_treatment)],
        'query hypothyroid': [int(query_hypo)],
        'query hyperthyroid': [int(query_hyper)],
        'lithium': [int(lithium)],
        'goitre': [int(goitre)],
        'tumor': [int(tumor)],
        'hypopituitary': [int(hypopituitary)],
        'psych': [int(psych)],
        'tsh': [tsh],
        't3': [t3],
        'tt4': [tt4],
        't4u': [t4u],
        'fti': [fti]
    }
    
    input_df = pd.DataFrame(input_data)

    # Reorder columns to match training data exactly
    # We intersect with feature_cols to ensure we have everything
    input_df = input_df[feature_cols]
    
    # Scale
    input_scaled = scaler.transform(input_df)
    
    # Predict
    pred_idx = knn.predict(input_scaled)[0]
    pred_class = le.inverse_transform([pred_idx])[0]
    
    st.subheader(f"Results for {name}")
    if pred_class == 'negative':
        st.success(f"Diagnosis: {pred_class}")
    else:
        st.error(f"Diagnosis: {pred_class}")
    st.info(solutions_map.get(pred_class, "Consult a doctor."))