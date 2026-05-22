import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load saved files ─────────────────────────────────────────────
model         = joblib.load('final_churn_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide"
)

# ── Sample customer profiles ─────────────────────────────────────
sample_profiles = {
    "Select a sample...": None,

    " High Risk Customer": {
        'SeniorCitizen': 0, 'Partner': 0, 'Dependents': 0,
        'tenure': 2, 'PhoneService': 1, 'Contract': 0,
        'PaperlessBilling': 1, 'MonthlyCharges': 95,
        'MultipleLines_No': 0, 'MultipleLines_Yes': 1,
        'InternetFiber': 1, 'InternetNo': 0,
        'OnlineSecurity_NI': 0, 'OnlineSecurity_Yes': 0,
        'OnlineBackup_NI': 0, 'OnlineBackup_Yes': 0,
        'DeviceProtection_NI': 0, 'DeviceProtection_Yes': 0,
        'TechSupport_NI': 0, 'TechSupport_Yes': 0,
        'StreamingTV_NI': 0, 'StreamingTV_Yes': 0,
        'StreamingMovies_NI': 0, 'StreamingMovies_Yes': 0,
        'PaymentCC': 0, 'PaymentEC': 1, 'PaymentMC': 0,
    },

    " Medium Risk Customer": {
        'SeniorCitizen': 0, 'Partner': 1, 'Dependents': 0,
        'tenure': 24, 'PhoneService': 1, 'Contract': 1,
        'PaperlessBilling': 1, 'MonthlyCharges': 65,
        'MultipleLines_No': 0, 'MultipleLines_Yes': 1,
        'InternetFiber': 1, 'InternetNo': 0,
        'OnlineSecurity_NI': 0, 'OnlineSecurity_Yes': 0,
        'OnlineBackup_NI': 0, 'OnlineBackup_Yes': 1,
        'DeviceProtection_NI': 0, 'DeviceProtection_Yes': 0,
        'TechSupport_NI': 0, 'TechSupport_Yes': 0,
        'StreamingTV_NI': 0, 'StreamingTV_Yes': 1,
        'StreamingMovies_NI': 0, 'StreamingMovies_Yes': 0,
        'PaymentCC': 0, 'PaymentEC': 0, 'PaymentMC': 1,
    },

    " Low Risk Customer": {
        'SeniorCitizen': 0, 'Partner': 1, 'Dependents': 1,
        'tenure': 60, 'PhoneService': 1, 'Contract': 2,
        'PaperlessBilling': 0, 'MonthlyCharges': 45,
        'MultipleLines_No': 0, 'MultipleLines_Yes': 1,
        'InternetFiber': 0, 'InternetNo': 0,
        'OnlineSecurity_NI': 0, 'OnlineSecurity_Yes': 1,
        'OnlineBackup_NI': 0, 'OnlineBackup_Yes': 1,
        'DeviceProtection_NI': 0, 'DeviceProtection_Yes': 1,
        'TechSupport_NI': 0, 'TechSupport_Yes': 1,
        'StreamingTV_NI': 0, 'StreamingTV_Yes': 1,
        'StreamingMovies_NI': 0, 'StreamingMovies_Yes': 1,
        'PaymentCC': 1, 'PaymentEC': 0, 'PaymentMC': 0,
    }
}

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.title(" Churn Predictor")
    st.markdown("---")

    st.subheader(" Sample Customers")
    st.caption("Load a preset customer profile to quickly demo the app")

    selected_sample = st.selectbox(
        "Choose a profile",
        options=list(sample_profiles.keys())
    )

    st.markdown("---")
    st.subheader(" About")
    st.markdown("""
    This app predicts whether a telecom customer
    is likely to churn using a **Logistic Regression**
    model trained on the Telco dataset.

    **Model Performance:**
    - AUC-ROC: 0.8323
    - Recall:  73.5%
    - F1:      0.6152
    """)

    st.markdown("---")
    st.caption("Built with Streamlit + SHAP")

# ── Get profile values if sample selected ────────────────────────
profile = sample_profiles[selected_sample]

def get_val(key, default):
    return profile[key] if profile else default

# ── Navigation ────────────────────────────────────────────────────
page = st.tabs([" Predict", " Model Performance"])

# ════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ════════════════════════════════════════════════════════════════
with page[0]:
    st.title(" Customer Churn Predictor")
    st.markdown("Fill in customer details or load a sample from the sidebar")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account Info")
        SeniorCitizen    = st.selectbox("Senior Citizen",
                               [0,1], index=get_val('SeniorCitizen', 0))
        Partner          = st.selectbox("Has Partner",
                               [0,1], index=get_val('Partner', 0))
        Dependents       = st.selectbox("Has Dependents",
                               [0,1], index=get_val('Dependents', 0))
        tenure           = st.slider("Tenure (months)", 0, 72,
                               get_val('tenure', 12))
        Contract         = st.selectbox("Contract Type",
                               options=[0,1,2],
                               index=get_val('Contract', 0),
                               format_func=lambda x:
                               {0:"Month-to-month",
                                1:"One year",
                                2:"Two year"}[x])
        PaperlessBilling = st.selectbox("Paperless Billing",
                               [0,1], index=get_val('PaperlessBilling', 0))

    with col2:
        st.subheader("Services")
        PhoneService       = st.selectbox("Phone Service",
                               [0,1], index=get_val('PhoneService', 0))
        MultipleLines_No   = st.selectbox("No Phone Service",
                               [0,1], index=get_val('MultipleLines_No', 0))
        MultipleLines_Yes  = st.selectbox("Multiple Lines",
                               [0,1], index=get_val('MultipleLines_Yes', 0))
        InternetFiber      = st.selectbox("Fiber Optic Internet",
                               [0,1], index=get_val('InternetFiber', 0))
        InternetNo         = st.selectbox("No Internet Service",
                               [0,1], index=get_val('InternetNo', 0))
        OnlineSecurity_NI  = st.selectbox("No Internet (Security)",
                               [0,1], index=get_val('OnlineSecurity_NI', 0))
        OnlineSecurity_Yes = st.selectbox("Online Security",
                               [0,1], index=get_val('OnlineSecurity_Yes', 0))

    with col3:
        st.subheader("Add-ons & Payment")
        OnlineBackup_NI     = st.selectbox("No Internet (Backup)",
                               [0,1], index=get_val('OnlineBackup_NI', 0))
        OnlineBackup_Yes    = st.selectbox("Online Backup",
                               [0,1], index=get_val('OnlineBackup_Yes', 0))
        DeviceProtection_NI = st.selectbox("No Internet (Device)",
                               [0,1], index=get_val('DeviceProtection_NI', 0))
        DeviceProtection_Yes= st.selectbox("Device Protection",
                               [0,1], index=get_val('DeviceProtection_Yes', 0))
        TechSupport_NI      = st.selectbox("No Internet (Support)",
                               [0,1], index=get_val('TechSupport_NI', 0))
        TechSupport_Yes     = st.selectbox("Tech Support",
                               [0,1], index=get_val('TechSupport_Yes', 0))
        StreamingTV_NI      = st.selectbox("No Internet (TV)",
                               [0,1], index=get_val('StreamingTV_NI', 0))
        StreamingTV_Yes     = st.selectbox("Streaming TV",
                               [0,1], index=get_val('StreamingTV_Yes', 0))
        StreamingMovies_NI  = st.selectbox("No Internet (Movies)",
                               [0,1], index=get_val('StreamingMovies_NI', 0))
        StreamingMovies_Yes = st.selectbox("Streaming Movies",
                               [0,1], index=get_val('StreamingMovies_Yes', 0))
        MonthlyCharges      = st.slider("Monthly Charges ($)", 0, 120,
                               get_val('MonthlyCharges', 65))
        PaymentCC           = st.selectbox("Credit Card Payment",
                               [0,1], index=get_val('PaymentCC', 0))
        PaymentEC           = st.selectbox("Electronic Check",
                               [0,1], index=get_val('PaymentEC', 0))
        PaymentMC           = st.selectbox("Mailed Check",
                               [0,1], index=get_val('PaymentMC', 0))

    st.divider()

    if st.button(" Predict Churn", use_container_width=True):

        input_data = pd.DataFrame([[
            SeniorCitizen, Partner, Dependents, tenure,
            PhoneService, Contract, PaperlessBilling, MonthlyCharges,
            MultipleLines_No, MultipleLines_Yes,
            InternetFiber, InternetNo,
            OnlineSecurity_NI, OnlineSecurity_Yes,
            OnlineBackup_NI, OnlineBackup_Yes,
            DeviceProtection_NI, DeviceProtection_Yes,
            TechSupport_NI, TechSupport_Yes,
            StreamingTV_NI, StreamingTV_Yes,
            StreamingMovies_NI, StreamingMovies_Yes,
            PaymentCC, PaymentEC, PaymentMC
        ]], columns=feature_names)

        # Scale manually
        input_data['tenure']         = (tenure - 32.37) / 24.56
        input_data['MonthlyCharges'] = (MonthlyCharges - 64.76) / 30.09

        probability = model.predict_proba(input_data)[0][1]
        prediction  = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        col_a, col_b = st.columns(2)

        with col_a:
            if prediction == 1:
                st.error(" High Churn Risk")
            else:
                st.success(" Low Churn Risk")

            st.metric("Churn Probability", f"{probability:.1%}")
            st.progress(float(probability))

        with col_b:
            st.metric("Retention Probability", f"{1-probability:.1%}")
            if probability > 0.7:
                st.warning(" Immediate retention action recommended")
            elif probability > 0.4:
                st.warning(" Monitor this customer closely")
            else:
                st.info(" Customer appears stable")

        st.divider()
        st.subheader("Why this prediction?")

        explainer   = shap.LinearExplainer(model, input_data)
        shap_values = explainer.shap_values(input_data)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_data.iloc[0],
                feature_names=feature_names
            ),
            show=False
        )
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════
with page[1]:
    st.title(" Model Performance")
    st.markdown("Final model: **Logistic Regression** — Telco Customer Churn dataset")
    st.divider()

    st.subheader("Final Model Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("AUC-ROC",   "0.8323")
    m2.metric("Accuracy",  "0.7559")
    m3.metric("Recall",    "0.7353")
    m4.metric("Precision", "0.5288")
    m5.metric("F1 Score",  "0.6152")

    st.divider()

    st.subheader("Model Comparison")
    st.caption("Logistic Regression selected as champion — lowest missed churners")

    comparison = pd.DataFrame({
        'Model':                [' Logistic Regression',
                                 'XGBoost Tuned',
                                 'XGBoost Default',
                                 'Random Forest'],
        'AUC-ROC':              [0.8323, 0.8327, 0.8244, 0.8206],
        'Recall':               [0.7353, 0.6595, 0.6542, 0.6300],
        'Precision':            [0.5288, 0.5503, 0.5495, 0.5609],
        'F1':                   [0.6152, 0.6000, 0.5973, 0.5934],
        'Missed Churners (FN)': [99,     127,    129,    138]
    })

    def highlight_best(s):
        if s.name in ['AUC-ROC', 'Recall', 'F1']:
            is_best = s == s.max()
        elif s.name == 'Missed Churners (FN)':
            is_best = s == s.min()
        else:
            return ['' for _ in s]
        return ['background-color: #d4edda; color: #155724'
                if v else '' for v in is_best]

    st.dataframe(
        comparison.style.apply(highlight_best),
        use_container_width=True,
        hide_index=True
    )
    st.caption(" Green = best value for that metric")

    st.divider()

