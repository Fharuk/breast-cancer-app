import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

# -------------------------------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------------------------------
st.set_page_config(page_title="Breast Cancer Prediction System", layout="wide")

# File Paths
MODEL_PATH = 'svm_breast_cancer_model_top.pkl'
SCALER_PATH = 'scaler_top_features.pkl'
FEATURES_PATH = 'top_features.pkl'
X_TRAIN_PATH = 'X_train_top_features.pkl'
LOG_FILE = "patient_predictions.csv"

# -------------------------------------------------------------------------------------------------
# CACHED RESOURCE LOADING
# -------------------------------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load models and setup SHAP explainer once to save performance."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Critical Error: Model file '{MODEL_PATH}' not found.")
        return None, None, None, None, None

    try:
        svm = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feats = joblib.load(FEATURES_PATH)
        x_train = joblib.load(X_TRAIN_PATH)
        
        # Initialize SHAP (Using small sample for speed)
        # We cache this because creating the explainer is expensive
        explainer = shap.KernelExplainer(svm.predict_proba, x_train.sample(50, random_state=42))
        
        return svm, scaler, feats, x_train, explainer
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None, None, None

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
    st.title("🏥 Breast Cancer Prediction System")
    st.markdown("### Professional Diagnostic Tool Support")

    # 1. Load Resources
    svm_model, scaler, top_features, X_train_top, explainer = load_artifacts()
    
    if svm_model is None:
        st.stop()

    # 2. Sidebar Inputs
    st.sidebar.header("Patient Measurements")
    
    input_data = {}
    
    # Dynamically generate sliders based on the features loaded from the pickle file
    # We group them for better UI organization if possible, otherwise list them
    for feature in top_features:
        # Determine logical min/max based on training data
        min_val = float(X_train_top[feature].min())
        max_val = float(X_train_top[feature].max())
        avg_val = float((min_val + max_val) / 2)
        
        input_data[feature] = st.sidebar.slider(
            label=feature,
            min_value=min_val,
            max_value=max_val,
            value=avg_val,
            step=0.01
        )

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # 3. Main Interface Tabs
    tab1, tab2, tab3 = st.tabs(["Prediction & Analysis", "SHAP Explanation", "Patient History"])

    with tab1:
        st.subheader("Diagnostic Assessment")
        
        if st.button("Run Analysis", type="primary"):
            # Scaling
            input_scaled = scaler.transform(input_df)
            
            # Prediction
            prediction = svm_model.predict(input_scaled)[0]
            prediction_proba = svm_model.predict_proba(input_scaled)[0][1]
            
            result_label = "Malignant" if prediction == 1 else "Benign"
            result_color = "red" if prediction == 1 else "green"
            
            # Display Result
            if prediction == 1:
                st.error(f"prediction: {result_label}")
            else:
                st.success(f"Prediction: {result_label}")
            
            st.metric(label="Probability of Malignancy", value=f"{prediction_proba:.2%}")
            
            # Log Data
            log_prediction(input_df, result_label, prediction_proba)
            st.toast("Result logged to database.")

            # Store in session state for SHAP tab access
            st.session_state['input_scaled'] = input_scaled
            st.session_state['run_shap'] = True

    with tab2:
        st.subheader("Feature Contribution Analysis (SHAP)")
        
        if 'run_shap' in st.session_state and st.session_state['run_shap']:
            with st.spinner("Calculating feature importance..."):
                input_scaled = st.session_state['input_scaled']
                shap_values = explainer.shap_values(input_scaled)[1] # Class 1
                
                # Sorting logic for plot
                feature_order = np.argsort(np.abs(shap_values[0]))[::-1]
                sorted_features = np.array(top_features)[feature_order]
                sorted_shap_values = shap_values[0][feature_order]

                # Plotting
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['red' if val > 0 else 'green' for val in sorted_shap_values]
                ax.barh(sorted_features, sorted_shap_values, color=colors)
                ax.set_xlabel("SHAP Value (Impact on Model Output)")
                ax.set_title("Feature Impact: Red pushes towards Malignant, Green towards Benign")
                plt.tight_layout()
                
                st.pyplot(fig)
        else:
            st.info("Run an analysis in the 'Prediction' tab to generate SHAP plots.")

    with tab3:
        st.subheader("Session History")
        if os.path.exists(LOG_FILE):
            log_df = pd.read_csv(LOG_FILE)
            st.dataframe(log_df.sort_values(by="Timestamp", ascending=False))
            
            csv = log_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download History CSV", csv, "patient_history.csv", "text/csv")
        else:
            st.write("No predictions logged yet.")

if __name__ == "__main__":
    main()