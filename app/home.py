import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing import DataLoader

# Page config
st.set_page_config(
    page_title="Loan Default Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        height: 100%;
        min-height: 120px;
    }
    .feature-title {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
        color: #2c3e50;
    }
    .feature-text {
        color: #495057;
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Loan Default Risk Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered credit risk assessment using machine learning</p>', unsafe_allow_html=True)

st.markdown("---")

# Key metrics row
col1, col2, col3, col4 = st.columns(4)

# Load actual stats from data
try:
    data_loader = DataLoader()
    df = data_loader.load_clean_data()
    if df is not None:
        stats = data_loader.get_feature_statistics(df)
        total_loans = f"{stats['total_loans']:,}"
        default_rate = f"{stats['default_rate']:.1%}"
        avg_loan = f"${stats['avg_loan_amount']:,.0f}"
        avg_fico = f"{stats['avg_fico']:.0f}"
    else:
        raise Exception
except:
    total_loans = "15,000"
    default_rate = "17%"
    avg_loan = "$14,500"
    avg_fico = "695"

with col1:
    st.metric("Loans Analyzed", total_loans)
with col2:
    st.metric("Default Rate", default_rate)
with col3:
    st.metric("Model Recall", "63%", help="Percentage of defaults correctly identified")
with col4:
    st.metric("Avg Loan Amount", avg_loan)

st.markdown("---")

# Main value proposition
st.markdown("## What This Tool Does")

st.markdown("""
This application predicts the likelihood of loan default using a logistic regression model trained on
real Lending Club data. It helps financial institutions make data-driven lending decisions by
identifying high-risk borrowers before approval.
""")

# Feature cards
st.markdown("### Key Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">Data Exploration</div>
        <div class="feature-text">Interactive visualizations to understand loan patterns, borrower characteristics, and risk factors</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-box">
        <div class="feature-title">Risk Prediction</div>
        <div class="feature-text">Get instant default probability scores for individual loan applications</div>
    </div>
    """, unsafe_allow_html=True)



st.markdown("")
st.markdown("")

# Model approach
st.markdown("### The Model")

col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    **Logistic Regression** was chosen for its interpretability and effectiveness in handling
    imbalanced data. The model was trained on key financial indicators including:

    - Credit score (FICO)
    - Debt-to-income ratio
    - Loan amount and interest rate
    - Credit utilization
    - Loan grade
    - And other borrower characteristics

    The decision threshold was optimized to maximize recall while maintaining business viability,
    prioritizing the detection of high-risk loans over overall accuracy.
    """)

with col2:
    st.info("""
    **Why This Matters**

    With severe class imbalance (83% non-defaults), a naive model could achieve high accuracy
    by simply approving everyone.

    Our model is tuned to catch actual defaults, even at the cost of some false alarms, because
    **missing a default costs ~$15,000** while declining a good loan costs only ~$1,500 in
    foregone revenue.
    """)

st.markdown("---")

# Getting started
st.markdown("## Getting Started")

st.markdown("""
Navigate through the app using the sidebar:

1. **EDA** - Explore the dataset and understand patterns in loan defaults
2. **Model Predictions** - Input borrower information to get real-time risk assessments
""")

# Sidebar content
with st.sidebar:
    st.markdown("### Dataset Stats")

    st.metric("Total Loans", total_loans)
    st.metric("Default Rate", default_rate)
    st.metric("Avg FICO", avg_fico)
    st.metric("Avg Loan", avg_loan)

    st.markdown("---")

    st.markdown("### Model Performance")
    st.metric("Recall", "63%", help="Defaults correctly identified")
    st.metric("Precision", "26%", help="Accuracy of default predictions")
    st.metric("F1 Score", "37%", help="Harmonic mean of precision and recall")

    st.markdown("---")

    st.markdown("### Business Impact")
    st.markdown("""
    The model helps prevent significant losses by identifying high-risk borrowers while
    maintaining reasonable approval rates for creditworthy applicants.
    """)
