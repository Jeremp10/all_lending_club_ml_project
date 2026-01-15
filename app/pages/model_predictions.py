import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model_utils import ModelPredictor, interpret_risk
from src.visualizations import LoanVisualizer

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(page_title="Loan Default Prediction", layout="wide")

st.title(" Loan Default Prediction")


# ----------------------------------
# Initialize utilities
# ----------------------------------
predictor = ModelPredictor()
visualizer = LoanVisualizer()

# Load model artifacts

if not predictor.load_model():
    st.error("""
    **Model artifacts not found**

    Required files in `models/`:
    - logistic_regression.pkl
    - scaler.pkl
    - feature_names.pkl
    - threshold.pkl
    """)
    st.stop()

st.success(" Model loaded successfully")

# ----------------------------------
# Input method
# ----------------------------------
st.markdown("---")
input_method = st.radio(
    "Choose input method:",
    [" Manual Input", " Example Cases"],
    horizontal=True
)

# ----------------------------------
# Example cases
# ----------------------------------
example_cases = {
    "Low Risk - Grade A": {
        "loan_amnt": 10000,
        "int_rate": 7.5,
        "annual_inc": 85000,
        "dti": 8.5,
        "fico_range_low": 750,
        "revol_util": 15.0,
        "loan_to_income": 0.12,
        "grade_encoded": 1
    },
    "Medium Risk - Grade C": {
        "loan_amnt": 15000,
        "int_rate": 12.0,
        "annual_inc": 55000,
        "dti": 18.5,
        "fico_range_low": 690,
        "revol_util": 45.0,
        "loan_to_income": 0.27,
        "grade_encoded": 3
    },
    "High Risk - Grade E": {
        "loan_amnt": 20000,
        "int_rate": 18.5,
        "annual_inc": 40000,
        "dti": 28.0,
        "fico_range_low": 650,
        "revol_util": 85.0,
        "loan_to_income": 0.50,
        "grade_encoded": 5
    }
}

# ----------------------------------
# Input handling
# ----------------------------------
if input_method == " Example Cases":
    st.markdown("### Pre-loaded Example Borrowers")
    selected_case = st.selectbox("Select an example:", list(example_cases.keys()))
    input_data = example_cases[selected_case]

else:
    st.markdown("### Enter Borrower Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        loan_amnt = st.number_input("Loan Amount ($)", 1000, 40000, 15000, 1000)
        int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0, 0.5)

        grade_input = st.selectbox("Loan Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'], index=2)
        grade_encoded = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}[grade_input]

    with col2:
        annual_inc = st.number_input("Annual Income ($)", 10000, 500000, 65000, 5000)
        fico_range_low = st.slider("FICO Score", 660, 850, 690, 5)
        dti = st.slider("Debt-to-Income Ratio (%)", 0.0, 50.0, 18.0, 0.5)

    with col3:
        revol_util = st.slider("Credit Utilization (%)", 0.0, 150.0, 50.0, 1.0)
        loan_to_income = loan_amnt / annual_inc if annual_inc > 0 else 0
        st.metric("Loan-to-Income Ratio", f"{loan_to_income:.2%}")

    input_data = {
        "loan_amnt": loan_amnt,
        "int_rate": int_rate,
        "annual_inc": annual_inc,
        "dti": dti,
        "fico_range_low": fico_range_low,
        "revol_util": revol_util,
        "loan_to_income": loan_to_income,
        "grade_encoded": grade_encoded
    }

# ----------------------------------
# Prediction
# ----------------------------------
st.markdown("---")

if st.button(" Predict Default Risk", type="primary", use_container_width=True):

    input_df = pd.DataFrame([input_data])

    try:
        result = predictor.predict(input_df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    probability = result["probability"]
    decision = result["prediction"]

    st.markdown("## Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        if decision == 1:
            st.error("###  HIGH RISK")
            st.markdown("**Likely to Default**")
        else:
            st.success("###  LOW RISK")
            st.markdown("**Likely to Repay**")

    with col2:
        st.metric("Default Probability", f"{probability:.1%}",
                 help="Model's estimated probability of default")

    with col3:
        st.metric("Decision Threshold", f"{result['threshold']:.1%}",
                 help="Optimized threshold for classification")
        margin = probability - result['threshold']
        st.caption(f"Margin: {margin:+.1%}")

    st.markdown("---")

    # Risk gauge visualization
    fig = visualizer.risk_gauge(probability, avg_rate=0.17)
    st.plotly_chart(fig, use_container_width=True)

    # Risk interpretation
    risk_info = interpret_risk(probability)

    if risk_info['level'] == "Very Low Risk":
        st.success(f"**{risk_info['level']}** — {risk_info['recommendation']}")
    elif risk_info['level'] == "Low Risk":
        st.info(f"**{risk_info['level']}** — {risk_info['recommendation']}")
    elif risk_info['level'] == "Medium Risk":
        st.warning(f"**{risk_info['level']}** — {risk_info['recommendation']}")
    else:
        st.error(f"**{risk_info['level']}** — {risk_info['recommendation']}")

    st.markdown("---")

    # Key risk factors
    st.markdown("##  Key Risk Factors")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Borrower Profile")
        profile_data = {
            "Metric": ["FICO Score", "Annual Income", "Loan Amount", "Loan Grade"],
            "Value": [
                f"{input_data['fico_range_low']:.0f}",
                f"${input_data['annual_inc']:,.0f}",
                f"${input_data['loan_amnt']:,.0f}",
                grade_input
            ]
        }
        st.dataframe(pd.DataFrame(profile_data), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("### Risk Indicators")
        risk_data = {
            "Metric": ["Debt-to-Income", "Credit Utilization", "Loan-to-Income", "Interest Rate"],
            "Value": [
                f"{input_data['dti']:.1f}%",
                f"{input_data['revol_util']:.1f}%",
                f"{input_data['loan_to_income']:.1%}",
                f"{input_data['int_rate']:.1f}%"
            ]
        }
        st.dataframe(pd.DataFrame(risk_data), hide_index=True, use_container_width=True)

    st.markdown("---")
