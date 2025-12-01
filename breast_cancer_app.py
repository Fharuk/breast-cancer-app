import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import numpy as np

# -------------------------------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Prediction System", layout="centered")

# File Paths (Notice X_train is removed)
MODEL_PATH = 'svm_breast_cancer_model_top.pkl'
SCALER_PATH = 'scaler_top_features.pkl'
FEATURES_PATH = 'top_features.pkl'
LOG_FILE = "patient_predictions.csv"

# -------------------------------------------------------------------------------------------------
# CACHED RESOURCE LOADING
# -------------------------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """
    Load only the essential model artifacts. 
    Removed X_train and SHAP to reduce file size and errors.
    """
    # Check for critical files
    required_files = [MODEL_PATH, SCALER_PATH, FEATURES_PATH]
    for f in required_files:
        if not os.path.exists(f):
            st.error(f"Critical Error: Missing file '{f}'.")
            return None, None, None

    try:
        svm = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feats = joblib.load(FEATURES_PATH)
        return svm, scaler, feats
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None

def log_prediction(input_df, prediction_label, probability):
    """Log the prediction to a local CSV file."""
    log_entry = input_df.copy()
    log_entry['Prediction'] = prediction_label
    log_entry['Probability'] = probability
    log_entry['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if os.path.exists(LOG_FILE):
        log_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        log_entry.to_csv(LOG_FILE, index=False)

# -------------------------------------------------------------------------------------------------
# MAIN APPLICATION
# -------------------------------------------------------------------------------------------------
def main():
    st.title("🏥 Breast Cancer Diagnostic Tool")
    st.markdown("### Clinical Support System")

    # 1. Load Resources
    svm_model, scaler, top_features = load_artifacts()
    
    if svm_model is None:
        st.stop()

    # 2. Input Section
    st.subheader("Patient Vitals & Measurements")
    st.info("Please enter the specific mean/worst/se values below.")
    
    input_data = {}
    
    # We use columns to organize the inputs neatly
    # Since we don't have X_train for min/max, we use number_input
    # which allows any valid number.
    col1, col2 = st.columns(2)
    
    for i, feature in enumerate(top_features):
        with (col1 if i % 2 == 0 else col2):
            # We set a default value of 0.0 to ensure float type
            input_data[feature] = st.number_input(
                label=feature,
                value=0.0,
                step=0.1,
                format="%.4f"
            )

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # 3. Prediction Action
    st.markdown("---")
    if st.button("Generate Diagnosis", type="primary"):
        try:
            # Scaling
            input_scaled = scaler.transform(input_df)
            
            # Prediction
            prediction = svm_model.predict(input_scaled)[0]
            prediction_proba = svm_model.predict_proba(input_scaled)[0][1]
            
            result_label = "Malignant" if prediction == 1 else "Benign"
            
            # Display Result
            if prediction == 1:
                st.error(f"### Diagnosis: {result_label}")
                st.warning(f"Confidence Level: {prediction_proba:.2%}")
            else:
                st.success(f"### Diagnosis: {result_label}")
                st.info(f"Confidence Level: {prediction_proba:.2%}")
            
            # Log Data
            log_prediction(input_df, result_label, prediction_proba)
            
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

    # 4. History Tab
    with st.expander("View Patient History Log"):
        if os.path.exists(LOG_FILE):
            log_df = pd.read_csv(LOG_FILE)
            st.dataframe(log_df.sort_values(by="Timestamp", ascending=False))
            
            csv = log_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download History CSV", csv, "patient_history.csv", "text/csv")
        else:
            st.write("No local history found.")

if __name__ == "__main__":
    main()