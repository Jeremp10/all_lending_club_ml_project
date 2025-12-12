import streamlit as st
import pandas as pd

# Page config
st.set_page_config(
    page_title="Loan Default Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header"> Loan Default Risk Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Credit Risk Assessment Tool</p>', unsafe_allow_html=True)

# Introduction
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Explore Data")
    st.write("Visualize loan patterns, distributions, and key risk factors")

with col2:
    st.markdown("### Predict Defaults")
    st.write("Use ML models to assess loan default probability")

with col3:
    st.markdown("### Understand Predictions")
    st.write("See which features drive default risk using SHAP")

st.markdown("---")

# Project Overview
st.markdown("## Project Overview")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### About This Tool

    This application uses machine learning to predict loan default risk based on Lending Club data.

    **Key Features:**
    - **Exploratory Data Analysis**: Interactive visualizations of 15,000+ loans
    - **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost
    - **Threshold Optimization**: Adjust decision threshold based on business costs
    - **Explainable AI**: SHAP values show which features drive predictions

    **Business Impact:**
    - Reduce default losses by identifying high-risk borrowers
    - Improve approval rates for creditworthy applicants
    - Data-driven risk assessment replacing manual review
    """)
    st.markdown("###  Model Selection Results")

    st.info("""
    **Challenge:** Severe class imbalance (83% non-defaults) made this a difficult prediction task.

    **Finding:** Traditional accuracy metrics are misleading. A model that predicts "No Default"
    for everyone achieves 83% accuracy but provides zero business value.
    """)

    # Comparison table
    results_df = pd.DataFrame({
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Accuracy': ['63%', '68%', '79% '],
        'Recall (Catch Defaults)': ['63% ', '50%', '16% '],
        'Business Value': ['Best', 'Medium', 'Poor'],
        'Recommendation': [' Use', ' Consider', ' Avoid']
    })

    st.dataframe(results_df, hide_index=True, use_container_width=True)

    st.success("""
    **Selected Model: Logistic Regression**

    While it has the lowest accuracy, it catches 4x more defaults than XGBoost,
    resulting in ~$11M less in losses per 10,000 loans.

    Key insight: In imbalanced classification, optimizing for the right metric
    (Recall for costly minority class) is more important than overall accuracy.
    """)


with col2:
    st.markdown("### Model Performance")


    st.metric("Best Model", "XGBoost", "")
    st.metric("ROC-AUC Score", "0.85", "+0.15 vs baseline")
    st.metric("Loans Analyzed", "15,000", "")
    st.metric("Features Used", "30+", "")

st.markdown("---")

# How to Use
st.markdown("##  How to Use This App")

st.markdown("""
1. ** EDA Page**: Start here to understand the data
   - View loan distributions
   - Explore default patterns
   - Identify risk factors

2. ** Model Predictions**: Make predictions
   - Enter borrower details
   - Get instant default probability
   - Compare multiple models

3. ** Threshold Tuning**: Optimize for your business
   - Adjust decision threshold
   - Balance false positives vs false negatives
   - See cost impact

4. ** Explainability**: Understand the "why"
   - See feature importance
   - SHAP values for individual predictions
   - Model transparency
""")

st.markdown("---")

#todo: add a little description here

# Sidebar
with st.sidebar:
    st.markdown("---")

    st.markdown("##  Quick Stats")
    st.metric("Dataset Size", "15,000 loans")
    st.metric("Default Rate", "17%")
    st.metric("Features", "30+")

    st.markdown("---")
    st.markdown("###  Model Metrics")
    st.success("ROC-AUC: 0.85")
    st.info("Precision: 0.72")
    st.warning("Recall: 0.68")
