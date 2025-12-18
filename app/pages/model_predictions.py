import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.model_utils import ModelPredictor, interpret_risk, calculate_expected_loss, calculate_business_metrics, get_feature_impact
from src.visualizations import LoanVisualizer

st.set_page_config(page_title="Model Predictions", layout="wide")

# Title
st.title(" Loan Default Prediction")
st.markdown("Enter borrower information to predict default probability")

# Initialize utilities
predictor = ModelPredictor()
visualizer = LoanVisualizer()

# Load models
st.markdown("### Loading Models...")
models, scaler, feature_names = predictor.load_models()

if models is None:
    st.error("""
     **Models not found!**

    Please train and save your models first by running the model saving code
    at the end of your modeling notebook (Notebook 04).

    Required files in `models/` directory:
    - logistic_regression.pkl
    - random_forest.pkl
    - xgboost.pkl
    - scaler.pkl
    - feature_names.pkl
    """)
    st.stop()

st.success(" Models loaded successfully!")

# Sidebar - Model Selection
st.sidebar.markdown("##  Model Selection")
selected_model = st.sidebar.radio(
    "Choose prediction model:",
    ["Logistic Regression (Recommended)", "Random Forest", "XGBoost"],
    help="Logistic Regression is recommended due to superior default detection (63% recall)"
)

# Map selection to model
model_map = {
    "Logistic Regression (Recommended)": ("logistic_regression", "Logistic Regression"),
    "Random Forest": ("random_forest", "Random Forest"),
    "XGBoost": ("xgboost", "XGBoost")
}
model_key, model_name = model_map[selected_model]

st.sidebar.markdown("---")
st.sidebar.markdown("###  Model Performance")

# Get metrics for selected model
metrics = predictor.get_model_metrics()
selected_metrics = metrics[model_key]

st.sidebar.metric("Accuracy", f"{selected_metrics['accuracy']:.1%}")
st.sidebar.metric("Recall", f"{selected_metrics['recall']:.1%}",
                 " Best" if model_key == 'logistic_regression' else "")
st.sidebar.metric("Precision", f"{selected_metrics['precision']:.1%}")
st.sidebar.metric("ROC-AUC", f"{selected_metrics['roc_auc']:.3f}")

if model_key == 'xgboost':
    st.sidebar.warning(" XGBoost has poor recall (16%) - misses most defaults!")

# Main content - Input method selection
st.markdown("---")
input_method = st.radio(
    "Choose input method:",
    [" Manual Input", " Example Cases"],
    horizontal=True
)

# Example cases
example_cases = {
    "Low Risk - Grade A": {
        "loan_amnt": 10000,
        "int_rate": 7.5,
        "grade_encoded": 1,
        "annual_inc": 85000,
        "dti": 8.5,
        "fico_range_low": 750,
        "revol_util": 15.0,
        "loan_to_income": 0.12
    },
    "Medium Risk - Grade C": {
        "loan_amnt": 15000,
        "int_rate": 12.0,
        "grade_encoded": 3,
        "annual_inc": 55000,
        "dti": 18.5,
        "fico_range_low": 690,
        "revol_util": 45.0,
        "loan_to_income": 0.27
    },
    "High Risk - Grade E": {
        "loan_amnt": 20000,
        "int_rate": 18.5,
        "grade_encoded": 5,
        "annual_inc": 40000,
        "dti": 28.0,
        "fico_range_low": 650,
        "revol_util": 85.0,
        "loan_to_income": 0.50
    }
}

if input_method == " Example Cases":
    st.markdown("### Pre-loaded Example Borrowers")

    selected_case = st.selectbox("Select an example:", list(example_cases.keys()))
    input_data = example_cases[selected_case]

    # Display example data
    st.info(f"**Selected: {selected_case}**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Loan Amount", f"${input_data['loan_amnt']:,.0f}")
        st.metric("Interest Rate", f"{input_data['int_rate']}%")
    with col2:
        st.metric("Annual Income", f"${input_data['annual_inc']:,.0f}")
        st.metric("FICO Score", f"{input_data['fico_range_low']:.0f}")
    with col3:
        st.metric("DTI Ratio", f"{input_data['dti']}%")
        st.metric("Credit Utilization", f"{input_data['revol_util']}%")
    with col4:
        grade_letters = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G'}
        st.metric("Grade", grade_letters.get(input_data['grade_encoded'], 'Unknown'))
        st.metric("Loan-to-Income", f"{input_data['loan_to_income']:.2%}")

else:  # Manual Input
    st.markdown("### Enter Borrower Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("####  Loan Details")
        loan_amnt = st.number_input(
            "Loan Amount ($)",
            min_value=1000,
            max_value=40000,
            value=15000,
            step=1000,
            help="Amount requested by borrower"
        )

        int_rate = st.slider(
            "Interest Rate (%)",
            min_value=5.0,
            max_value=30.0,
            value=12.0,
            step=0.5,
            help="APR assigned to the loan"
        )

        grade_input = st.selectbox(
            "Loan Grade",
            options=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            index=2,  # Default to C
            help="Risk grade assigned (A=best, G=worst)"
        )
        grade_encoded = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}[grade_input]

    with col2:
        st.markdown("####  Borrower Profile")
        annual_inc = st.number_input(
            "Annual Income ($)",
            min_value=10000,
            max_value=500000,
            value=65000,
            step=5000,
            help="Borrower's yearly income"
        )

        fico_range_low = st.slider(
            "FICO Score",
            min_value=660,
            max_value=850,
            value=690,
            step=5,
            help="Credit score (660-850)"
        )

        dti = st.slider(
            "Debt-to-Income Ratio (%)",
            min_value=0.0,
            max_value=50.0,
            value=18.0,
            step=0.5,
            help="Monthly debt payments / Monthly income"
        )

    with col3:
        st.markdown("####  Credit History")
        revol_util = st.slider(
            "Credit Utilization (%)",
            min_value=0.0,
            max_value=150.0,
            value=50.0,
            step=1.0,
            help="Revolving credit used / Total available"
        )

        # Auto-calculate loan-to-income
        loan_to_income = loan_amnt / annual_inc
        st.metric(
            "Loan-to-Income Ratio",
            f"{loan_to_income:.2%}",
            help="Calculated: Loan Amount / Annual Income"
        )

        # Quick assessment
        fico_assessment = "Excellent" if fico_range_low >= 740 else "Good" if fico_range_low >= 670 else "Fair"
        dti_assessment = "Low" if dti < 20 else "Moderate" if dti < 35 else "High"
        util_assessment = "Good" if revol_util < 30 else "Fair" if revol_util < 70 else "High"

        st.info(f"""
        **Quick Assessment:**
        - FICO {fico_range_low}: {fico_assessment}
        - DTI {dti}%: {dti_assessment}
        - Utilization {revol_util}%: {util_assessment}
        """)

    # Store manual input
    input_data = {
        "loan_amnt": loan_amnt,
        "int_rate": int_rate,
        "grade_encoded": grade_encoded,
        "annual_inc": annual_inc,
        "dti": dti,
        "fico_range_low": fico_range_low,
        "revol_util": revol_util,
        "loan_to_income": loan_to_income
    }

# Predict button
st.markdown("---")

if st.button("🔮 Predict Default Risk", type="primary", use_container_width=True):

    st.markdown("###  Prediction Results")

    # Warning about simplified features
    st.warning("""
     **Note:** This demo uses 8 core numerical features for prediction.
    The full production model would include all one-hot encoded categorical variables
    (loan purpose, home ownership, employment length, etc.).
    """)

    try:
        # Create feature DataFrame (simplified - just the 8 core features)
        input_df = pd.DataFrame([input_data])

        # Get prediction from selected model
        result = predictor.predict(input_df, model_key)

        if result is None:
            st.error("Prediction failed. Please check model files.")
            st.stop()

        default_probability = result['probability']
        prediction = result['prediction']

        # Display main results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("###  Prediction")
            if prediction == 1:
                st.error("**HIGH RISK**")
                st.markdown("**Likely to Default**")
            else:
                st.success("**LOW RISK**")
                st.markdown("**Likely to Repay**")

        with col2:
            st.markdown("###  Default Probability")
            st.metric(
                "Probability of Default",
                f"{default_probability:.1%}",
                delta=f"{default_probability - 0.17:.1%} vs avg (17%)",
                delta_color="inverse"
            )

        with col3:
            st.markdown("###  Expected Loss")
            expected_loss = calculate_expected_loss(input_data['loan_amnt'], default_probability)
            st.metric(
                "Expected Loss",
                f"${expected_loss:,.0f}",
                help="Loan Amount × Default Probability × (1 - Recovery Rate)"
            )

        st.markdown("---")

        # Risk gauge
        st.markdown("###  Risk Gauge")
        fig = visualizer.risk_gauge(default_probability, avg_rate=0.17)
        st.plotly_chart(fig, use_container_width=True)

        # Risk interpretation
        st.markdown("###  Risk Assessment")

        risk_info = interpret_risk(default_probability)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"**Risk Level:** {risk_info['emoji']} **{risk_info['level']}**")

        with col2:
            st.info(risk_info['recommendation'])

        # Business metrics
        st.markdown("###  Business Analysis")

        business_metrics = calculate_business_metrics(
            input_data['loan_amnt'],
            input_data['int_rate'] / 100,  # Convert to decimal
            default_probability
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Expected Revenue", f"${business_metrics['expected_revenue']:,.0f}")
        with col2:
            st.metric("Expected Loss", f"${business_metrics['expected_loss']:,.0f}")
        with col3:
            net_value = business_metrics['net_expected_value']
            st.metric(
                "Net Expected Value",
                f"${net_value:,.0f}",
                delta="Profitable" if net_value > 0 else "Unprofitable"
            )
        with col4:
            st.metric(
                "Break-Even Prob",
                f"{business_metrics['break_even_probability']:.1%}",
                help="Max default probability to remain profitable"
            )

        if net_value > 0:
            st.success(f" **Business Recommendation: {business_metrics['recommendation']}** - Positive expected value of ${net_value:,.0f}")
        else:
            st.error(f" **Business Recommendation: {business_metrics['recommendation']}** - Negative expected value of ${net_value:,.0f}")

        # Feature impact analysis
        st.markdown("###  Key Risk Factors")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Positive Factors (Reduce Risk):**")

            # FICO
            fico_impact = get_feature_impact(input_data['fico_range_low'], 'fico_range_low')
            if fico_impact['impact'] == 'positive':
                st.success(f" {fico_impact['description']} ({input_data['fico_range_low']:.0f})")

            # DTI
            dti_impact = get_feature_impact(input_data['dti'], 'dti')
            if dti_impact['impact'] == 'positive':
                st.success(f" {dti_impact['description']} ({input_data['dti']:.1f}%)")

            # Credit Utilization
            util_impact = get_feature_impact(input_data['revol_util'], 'revol_util')
            if util_impact['impact'] == 'positive':
                st.success(f" {util_impact['description']} ({input_data['revol_util']:.1f}%)")

            # Loan to Income
            lti_impact = get_feature_impact(input_data['loan_to_income'], 'loan_to_income')
            if lti_impact['impact'] == 'positive':
                st.success(f" {lti_impact['description']} ({input_data['loan_to_income']:.1%})")

        with col2:
            st.markdown("**Risk Factors (Increase Risk):**")

            # FICO
            if fico_impact['impact'] == 'negative':
                st.error(f" {fico_impact['description']} ({input_data['fico_range_low']:.0f})")

            # DTI
            if dti_impact['impact'] == 'negative':
                st.error(f" {dti_impact['description']} ({input_data['dti']:.1f}%)")

            # Credit Utilization
            if util_impact['impact'] == 'negative':
                st.error(f" {util_impact['description']} ({input_data['revol_util']:.1f}%)")

            # Loan to Income
            if lti_impact['impact'] == 'negative':
                st.error(f" {lti_impact['description']} ({input_data['loan_to_income']:.1%})")

            # Interest Rate
            rate_impact = get_feature_impact(input_data['int_rate'], 'int_rate')
            if rate_impact['impact'] == 'negative':
                st.error(f" {rate_impact['description']} ({input_data['int_rate']:.1f}%)")

    except Exception as e:
        st.error(f"""
         **Prediction Error**

        {str(e)}

        This might be due to:
        - Feature mismatch between input and training data
        - Model files corrupted
        - Scaler not compatible

        Please check your model files and try again.
        """)
        import traceback
        with st.expander(" Debug Info"):
            st.code(traceback.format_exc())

# Model comparison section
st.markdown("---")
with st.expander(" Compare All Models"):
    st.markdown("### Model Comparison for This Borrower")

    st.info("See how different models assess this borrower's risk")

    try:
        input_df = pd.DataFrame([input_data])

        # Get predictions from all models
        all_results = predictor.predict_all_models(input_df)

        if all_results:
            # Create comparison dataframe
            comparison_data = []
            for model_key, result in all_results.items():
                model_metrics = predictor.get_model_metrics()[model_key]
                comparison_data.append({
                    'Model': model_metrics['name'],
                    'Default Probability': f"{result['probability']:.1%}",
                    'Prediction': 'Default' if result['prediction'] == 1 else 'No Default',
                    'Model Recall': f"{model_metrics['recall']:.1%}",
                    'Model Accuracy': f"{model_metrics['accuracy']:.1%}"
                })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, hide_index=True, use_container_width=True)

            # Visual comparison
            st.markdown("### Visual Comparison")
            fig = visualizer.model_comparison_bar(all_results)
            st.plotly_chart(fig, use_container_width=True)

            # Analysis
            probabilities = [r['probability'] for r in all_results.values()]
            spread = max(probabilities) - min(probabilities)

            if spread > 0.20:
                st.warning(f"""
                 **High Model Disagreement**: Models differ by {spread:.1%} in their predictions.

                This suggests:
                - Borrower profile is challenging to assess
                - Some features strongly influence certain models
                - Consider additional verification or stricter terms
                """)
            else:
                st.success(f" **Models Agree**: All models predict similar risk levels (spread: {spread:.1%})")

    except Exception as e:
        st.error(f"Could not compare models: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
###  Tips for Using This Tool

- **Start with example cases** to understand how different risk profiles are assessed
- **Compare all models** to see how they differ in their predictions
- **Focus on business metrics** (Net Expected Value) not just probability
- **Remember**: This is a simplified demo - production systems would include more features and validation
""")
