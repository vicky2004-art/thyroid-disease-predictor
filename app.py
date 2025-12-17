import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Page Configuration
st.set_page_config(page_title="Thyroid Disease Prediction", layout="wide")

# --- 1. Load and Preprocess Data ---
@st.cache_resource
def load_and_train_model():
    # Load dataset
    try:
        df = pd.read_csv('Thyroid-Dataset.csv')
    except FileNotFoundError:
        st.error("CSV file 'Thyroid-Dataset.csv' not found. Please place it in the same directory.")
        return None, None, None, None

    # Drop irrelevant columns
    if 'referral source' in df.columns:
        df = df.drop(['referral source'], axis=1)

    # Clean 'sex' column (Map F/M to 0/1 and fill NaNs)
    df['sex'] = df['sex'].map({'F': 0, 'M': 1})
    df['sex'] = df['sex'].fillna(df['sex'].mode()[0])

    # Convert Boolean columns to Integers (0 and 1)
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    # Handle Missing Values in Numeric Columns (Impute with Median)
    num_cols = ['TSH', 'T3', 'TT4', 'T4U', 'FTI']
    # Coerce errors to NaN in case of non-numeric strings, then impute
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])

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

# Load model components
knn, scaler, le, feature_names = load_and_train_model()

# Solution Mapping Dictionary
solutions_map = {
    'negative': "No thyroid disease detected. Maintain a healthy lifestyle and regular checkups.",
    'hypothyroid conditions': "Possible Hypothyroidism. Consult an endocrinologist. Common treatments include hormone replacement (e.g., Levothyroxine).",
    'general health': "General health check recommended. No specific thyroid disorder strongly indicated, but consult a doctor if symptoms persist.",
    'binding protein': "Abnormal binding protein levels detected. This may affect hormone transport. Consult a specialist for detailed analysis.",
    'replacement therapy': "Patient appears to be on hormone replacement therapy. Continue monitoring hormone levels regularly.",
    'discordant results': "Lab results are inconsistent. It is highly recommended to repeat the tests for a confirmed diagnosis.",
    'hyperthyroid conditions': "Possible Hyperthyroidism. Consult an endocrinologist. Treatments may include antithyroid medication (e.g., Methimazole) or radioactive iodine.",
    'miscellaneous': "Condition classified as miscellaneous. A detailed clinical evaluation is needed to determine the specific issue.",
    'antithyroid treatment': "Patient appears to be on antithyroid treatment. Continue monitoring treatment efficacy with your doctor."
}

# --- 2. Website UI ---
st.title("🏥 Thyroid Disease Prediction System")
st.markdown("This system uses the **K-Nearest Neighbors (KNN)** algorithm to predict thyroid disease based on patient symptoms and lab results.")

# Create a form for user input
with st.form("patient_form"):
    st.header("Patient Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        name = st.text_input("Patient Name")
    with col2:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
    with col3:
        sex_input = st.selectbox("Gender", ["Female", "Male"])
        sex = 0 if sex_input == "Female" else 1

    st.header("Symptoms & History")
    col_sym1, col_sym2, col_sym3 = st.columns(3)
    
    with col_sym1:
        sick = st.checkbox("Sick / Ill")
        pregnant = st.checkbox("Pregnant")
        thyroid_surgery = st.checkbox("History of Thyroid Surgery")
        goitre = st.checkbox("Goitre (Swollen Neck)")
        tumor = st.checkbox("Tumor")
        
    with col_sym2:
        on_thyroxine = st.checkbox("On Thyroxine Medication")
        query_on_thyroxine = st.checkbox("Query: On Thyroxine")
        on_antithyroid = st.checkbox("On Antithyroid Medication")
        I131_treatment = st.checkbox("I131 Treatment History")
        
    with col_sym3:
        query_hypo = st.checkbox("Query: Hypothyroid")
        query_hyper = st.checkbox("Query: Hyperthyroid")
        lithium = st.checkbox("On Lithium")
        hypopituitary = st.checkbox("Hypopituitary")
        psych = st.checkbox("Psychological Symptoms")

    st.header("Lab Results (Leave default if unknown)")
    st.caption("Default values are set to population medians.")
    
    # Defaults based on approximate medians from dataset to avoid skewing if left blank
    col_lab1, col_lab2, col_lab3, col_lab4, col_lab5 = st.columns(5)
    with col_lab1:
        tsh = st.number_input("TSH", value=1.4)
    with col_lab2:
        t3 = st.number_input("T3", value=2.0)
    with col_lab3:
        tt4 = st.number_input("TT4", value=103.0)
    with col_lab4:
        t4u = st.number_input("T4U", value=0.98)
    with col_lab5:
        fti = st.number_input("FTI", value=107.0)

    submit_button = st.form_submit_button("Predict Disease")

# --- 3. Prediction Logic ---
if submit_button:
    if knn is not None:
        # Prepare input vector in the exact order of training columns
        # Columns: age, sex, on thyroxine, query on thyroxine, on antithyroid medication, sick, pregnant,
        # thyroid surgery, I131 treatment, query hypothyroid, query hyperthyroid, lithium, goitre, tumor,
        # hypopituitary, psych, TSH, T3, TT4, T4U, FTI
        
        input_data = [
            age, sex, 
            int(on_thyroxine), int(query_on_thyroxine), int(on_antithyroid), 
            int(sick), int(pregnant), int(thyroid_surgery), int(I131_treatment),
            int(query_hypo), int(query_hyper), int(lithium), int(goitre), 
            int(tumor), int(hypopituitary), int(psych),
            tsh, t3, tt4, t4u, fti
        ]
        
        # Scale the input
        input_data_scaled = scaler.transform([input_data])
        
        # Predict
        prediction_index = knn.predict(input_data_scaled)[0]
        prediction_class = le.inverse_transform([prediction_index])[0]
        solution = solutions_map.get(prediction_class, "Consult a specialist for further advice.")

        # Display Results
        st.divider()
        st.subheader(f"Results for {name}")
        
        # Color code the result
        if prediction_class == 'negative':
            st.success(f"**Predicted Diagnosis:** {prediction_class.title()}")
        elif 'hypo' in prediction_class or 'hyper' in prediction_class:
            st.error(f"**Predicted Diagnosis:** {prediction_class.title()}")
        else:
            st.warning(f"**Predicted Diagnosis:** {prediction_class.title()}")
            
        st.info(f"**Suggested Solution / Advice:**\n\n{solution}")
    else:
        st.error("Model failed to load. Please check the CSV file.")