"""
Model loading and prediction utilities

"""

import pickle
from pathlib import Path
import pandas as pd
import streamlit as st


class ModelPredictor:
    """
    Handles loading the trained model and making predictions.

    """

    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.threshold = None

    def load_model(self):
        """
        Load all model artifacts.
        """
        try:
            with open(self.models_dir / "logistic_regression.pkl", "rb") as f:
                self.model = pickle.load(f)

            with open(self.models_dir / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)

            with open(self.models_dir / "feature_names.pkl", "rb") as f:
                self.feature_names = pickle.load(f)

            with open(self.models_dir / "threshold.pkl", "rb") as f:
                self.threshold = pickle.load(f)

            return True

        except FileNotFoundError as e:
            st.error(f"Missing model artifact: {e}")
            return False

    def predict(self, input_data: pd.DataFrame):
        """
        Predict default probability and decision.
        """
        if self.model is None:
            if not self.load_model():
                return None

        # Debug: Check if feature_names loaded properly
        if self.feature_names is None:
            raise ValueError("Feature names not loaded. Please check model artifacts.")

        # Enforce feature order
        input_data = input_data[self.feature_names]

        # Scale
        input_scaled = self.scaler.transform(input_data)

        # Probability of default
        probability = self.model.predict_proba(input_scaled)[0, 1]

        # Decision using optimized threshold
        decision = int(probability >= self.threshold)

        return {
            "prediction": decision,
            "probability": float(probability),
            "threshold": float(self.threshold),
            "margin": float(probability - self.threshold)
        }


# -----------------------------
# Business / interpretation utils
# -----------------------------

def interpret_risk(probability: float):
    """
    Interpret risk bands from calibrated probability.
    """
    if probability < 0.10:
        return {
            "level": "Very Low Risk",
            "color": "green",
            "recommendation": "Approve — very strong borrower"
        }
    elif probability < 0.20:
        return {
            "level": "Low Risk",
            "color": "blue",
            "recommendation": "Approve — below average risk"
        }
    elif probability < 0.40:
        return {
            "level": "Medium Risk",
            "color": "orange",
            "recommendation": "Review — adjust pricing or terms"
        }
    elif probability < 0.60:
        return {
            "level": "High Risk",
            "color": "red",
            "recommendation": "Caution — high default risk"
        }
    else:
        return {
            "level": "Very High Risk",
            "color": "darkred",
            "recommendation": "Decline — unacceptable risk"
        }


def calculate_expected_loss(loan_amount, default_probability, recovery_rate=0.10):
    return loan_amount * default_probability * (1 - recovery_rate)


def calculate_business_metrics(
    loan_amount,
    interest_rate,
    default_probability,
    recovery_rate=0.10
):
    """
    Calculate business metrics for a loan.

    Expected revenue is the interest earned IF the loan is repaid (weighted by probability).
    Expected loss is the principal lost IF the loan defaults (weighted by probability).
    Net expected value is the difference between the two.
    """
    # Expected revenue: interest earned when loan is repaid (probability-weighted)
    expected_revenue = loan_amount * interest_rate * (1 - default_probability)

    # Expected loss: principal lost when loan defaults (probability-weighted)
    expected_loss = calculate_expected_loss(
        loan_amount, default_probability, recovery_rate
    )

    # Net expected value: expected profit/loss per loan
    net_expected_value = expected_revenue - expected_loss

    # Break-even probability: the default rate at which we break even
    # Interest earned = Loss from default
    # loan_amount × interest_rate = loan_amount × p_default × (1 - recovery_rate)
    # p_default = interest_rate / (1 - recovery_rate)
    break_even_prob = interest_rate / (1 - recovery_rate)

    return {
        "expected_revenue": expected_revenue,
        "expected_loss": expected_loss,
        "net_expected_value": net_expected_value,
        "break_even_probability": break_even_prob,
        "recommendation": "Approve" if net_expected_value > 0 else "Decline"
    }
