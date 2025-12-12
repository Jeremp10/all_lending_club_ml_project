import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Model Predictions", page_icon="🤖", layout="wide")

# Title
st.title(" Loan Default Prediction")
st.markdown("Enter borrower information to predict default probability")

# Load models and scaler
@st.cache_resource
def load_models():
    """Load trained models and scaler"""
    try:
        with open('models/logistic_regression.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        with open('models/random_forest.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        with open('models/xgboost.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)

        return lr_model, rf_model, xgb_model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"""
         **Models not found!**

        Please make sure you've saved your trained models by running the modeling notebook
        and executing the model saving code.

        Error: {e}
        """)
        return None, None, None, None, None

# Load models
models_loaded = load_models()
if models_loaded[0] is None:
    st.stop()

lr_model, rf_model, xgb_model, scaler, feature_names = models_loaded

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
    "Logistic Regression (Recommended)": ("Logistic Regression", lr_model),
    "Random Forest": ("Random Forest", rf_model),
    "XGBoost": ("XGBoost", xgb_model)
}
model_name, model = model_map[selected_model]

st.sidebar.markdown("---")
st.sidebar.markdown("###  Model Stats")
if model_name == "Logistic Regression":
    st.sidebar.metric("Accuracy", "63%")
    st.sidebar.metric("Recall", "63%", " Best")
    st.sidebar.metric("Precision", "26%")
elif model_name == "Random Forest":
    st.sidebar.metric("Accuracy", "68%")
    st.sidebar.metric("Recall", "50%")
    st.sidebar.metric("Precision", "27%")
else:  # XGBoost
    st.sidebar.metric("Accuracy", "79%")
    st.sidebar.metric("Recall", "16%", " Poor")
    st.sidebar.metric("Precision", "31%")

# Main content - Two options: Manual Input or Example Cases
st.markdown("---")
input_method = st.radio(
    "Choose input method:",
    [" Manual Input", " Example Cases"],
    horizontal=True
)

if input_method == " Example Cases":
    st.markdown("### Pre-loaded Example Borrowers")

    example_cases = {
        "Low Risk - Grade A": {
            "loan_amnt": 10000,
            "int_rate": 7.5,
            "grade_encoded": 1,  # A
            "annual_inc": 85000,
            "dti": 8.5,
            "fico_range_low": 750,
            "revol_util": 15.0,
            "loan_to_income": 0.12
        },
        "Medium Risk - Grade C": {
            "loan_amnt": 15000,
            "int_rate": 12.0,
            "grade_encoded": 3,  # C
            "annual_inc": 55000,
            "dti": 18.5,
            "fico_range_low": 690,
            "revol_util": 45.0,
            "loan_to_income": 0.27
        },
        "High Risk - Grade E": {
            "loan_amnt": 20000,
            "int_rate": 18.5,
            "grade_encoded": 5,  # E
            "annual_inc": 40000,
            "dti": 28.0,
            "fico_range_low": 650,
            "revol_util": 85.0,
            "loan_to_income": 0.50
        }
    }

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

        st.info(f"""
        **Quick Assessment:**
        - FICO {fico_range_low}: {"Excellent" if fico_range_low >= 740 else "Good" if fico_range_low >= 670 else "Fair"}
        - DTI {dti}%: {"Low" if dti < 20 else "Moderate" if dti < 35 else "High"}
        - Utilization {revol_util}%: {"Good" if revol_util < 30 else "Fair" if revol_util < 70 else "High"}
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
if st.button(" Predict Default Risk", type="primary", use_container_width=True):

    # Create feature vector matching training data
    # Note: This is simplified - you'll need to match ALL features from training
    # Including one-hot encoded columns for purpose, home_ownership, emp_length

    st.markdown("###  Prediction Results")

    # For now, using just the numerical features
    # TODO: Add proper feature engineering with all encoded columns

    # Create base features DataFrame
    input_df = pd.DataFrame([input_data])

    # Add dummy columns for categorical features (simplified version)
    # In production, you'd need to match exact feature_names from training

    # Warning about simplified prediction
    st.warning("""
     **Simplified Prediction Mode**

    This demo uses core features only. Full production model would include:
    - Loan purpose (debt consolidation, credit card, etc.)
    - Home ownership status
    - Employment length
    - Additional engineered features
    """)

    try:
        # Scale features
        input_scaled = scaler.transform(input_df)

        # Get prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        default_probability = prediction_proba[1]  # Probability of default

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("###  Prediction")
            if prediction == 1:
                st.error("**HIGH RISK - Likely to Default**")
            else:
                st.success("**LOW RISK - Likely to Repay**")

        with col2:
            st.markdown("###  Default Probability")
            st.metric(
                "Probability of Default",
                f"{default_probability:.1%}",
                delta=f"{default_probability - 0.17:.1%} vs avg",
                delta_color="inverse"
            )

        with col3:
            st.markdown("###  Expected Loss")
            expected_loss = input_data['loan_amnt'] * default_probability
            st.metric(
                "Expected Loss",
                f"${expected_loss:,.0f}",
                help="Loan Amount × Default Probability"
            )

        # Probability gauge
        st.markdown("###  Risk Gauge")

        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = default_probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Default Risk (%)", 'font': {'size': 24}},
            delta = {'reference': 17, 'suffix': '% vs avg'},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 10], 'color': '#d4edda'},   # Very Low Risk
                    {'range': [10, 20], 'color': '#fff3cd'},  # Low Risk
                    {'range': [20, 40], 'color': '#ffe69c'},  # Medium Risk
                    {'range': [40, 60], 'color': '#f8d7da'},  # High Risk
                    {'range': [60, 100], 'color': '#f5c6cb'}  # Very High Risk
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 17  # Average default rate
                }
            }
        ))

        fig.update_layout(
            paper_bgcolor = "white",
            font = {'color': "darkblue", 'family': "Arial"},
            height = 400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Risk interpretation
        st.markdown("###  Risk Interpretation")

        if default_probability < 0.10:
            risk_level = "Very Low Risk"
            color = "green"
            recommendation = " **Approve** - Strong candidate with minimal default risk"
        elif default_probability < 0.20:
            risk_level = "Low Risk"
            color = "blue"
            recommendation = " **Approve** - Below average default risk"
        elif default_probability < 0.40:
            risk_level = "Medium Risk"
            color = "orange"
            recommendation = " **Review** - Above average risk, consider terms adjustment"
        elif default_probability < 0.60:
            risk_level = "High Risk"
            color = "red"
            recommendation = " **Caution** - High default probability, stricter terms required"
        else:
            risk_level = "Very High Risk"
            color = "darkred"
            recommendation = " **Decline** - Unacceptable default risk"

        st.markdown(f"**Risk Level:** :{color}[{risk_level}]")
        st.info(recommendation)

        # Feature contribution (simplified)
        st.markdown("###  Key Risk Factors")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Positive Factors (Reduce Risk):**")
            if fico_range_low >= 740:
                st.success(f" Excellent FICO score ({fico_range_low})")
            if dti < 15:
                st.success(f" Low debt burden ({dti}% DTI)")
            if revol_util < 30:
                st.success(f" Low credit utilization ({revol_util}%)")
            if loan_to_income < 0.20:
                st.success(f" Affordable loan size ({loan_to_income:.1%} of income)")

        with col2:
            st.markdown("**Risk Factors (Increase Risk):**")
            if fico_range_low < 680:
                st.error(f" Below average FICO ({fico_range_low})")
            if dti > 25:
                st.error(f" High debt burden ({dti}% DTI)")
            if revol_util > 70:
                st.error(f" High credit utilization ({revol_util}%)")
            if loan_to_income > 0.35:
                st.error(f" Large loan relative to income ({loan_to_income:.1%})")
            if int_rate > 15:
                st.error(f" High interest rate ({int_rate}%)")

    except Exception as e:
        st.error(f"""
         **Prediction Error**

        {str(e)}

        This might be due to feature mismatch. Make sure:
        1. Models were trained with same features
        2. Scaler was saved correctly
        3. Feature names match training data
        """)

# Model comparison
st.markdown("---")
with st.expander(" Compare All Models"):
    st.markdown("### Model Comparison for This Borrower")

    st.info("Compare predictions across all three models")

    try:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        # Get predictions from all models
        lr_prob = lr_model.predict_proba(input_scaled)[0][1]
        rf_prob = rf_model.predict_proba(input_scaled)[0][1]
        xgb_prob = xgb_model.predict_proba(input_scaled)[0][1]

        comparison_df = pd.DataFrame({
            'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
            'Default Probability': [f"{lr_prob:.1%}", f"{rf_prob:.1%}", f"{xgb_prob:.1%}"],
            'Prediction': [
                'Default' if lr_prob >= 0.5 else 'No Default',
                'Default' if rf_prob >= 0.5 else 'No Default',
                'Default' if xgb_prob >= 0.5 else 'No Default'
            ],
            'Model Recall': ['63%', '50%', '16%']
        })

        st.dataframe(comparison_df, hide_index=True, use_container_width=True)

        # Visual comparison
        fig = go.Figure(data=[
            go.Bar(
                name='Default Probability',
                x=['Logistic Regression', 'Random Forest', 'XGBoost'],
                y=[lr_prob * 100, rf_prob * 100, xgb_prob * 100],
                text=[f"{lr_prob:.1%}", f"{rf_prob:.1%}", f"{xgb_prob:.1%}"],
                textposition='outside',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
            )
        ])

        fig.add_hline(y=17, line_dash="dash", line_color="red",
                     annotation_text="Average Default Rate (17%)")

        fig.update_layout(
            title="Predicted Default Probability by Model",
            yaxis_title="Default Probability (%)",
            showlegend=False,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Could not compare models: {str(e)}")
