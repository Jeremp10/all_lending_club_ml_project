"""Model loading and prediction utilities"""
import pickle
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np

class ModelPredictor:
    """Handle model loading and predictions"""

    def __init__(self, models_dir='models'):
        self.models_dir = Path(models_dir)
        self.models = None
        self.scaler = None
        self.feature_names = None

    @st.cache_resource
    def load_models(_self):
        """Load all trained models"""
        try:
            models = {}
            model_files = {
                'logistic_regression': 'logistic_regression.pkl',
                'random_forest': 'random_forest.pkl',
                'xgboost': 'xgboost.pkl'
            }

            for name, filename in model_files.items():
                with open(_self.models_dir / filename, 'rb') as f:
                    models[name] = pickle.load(f)

            # Load scaler
            with open(_self.models_dir / 'scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)

            # Load feature names
            with open(_self.models_dir / 'feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)

            _self.models = models
            _self.scaler = scaler
            _self.feature_names = feature_names

            return models, scaler, feature_names

        except FileNotFoundError as e:
            st.error(f"Model files not found: {e}")
            return None, None, None

    def predict(self, input_data, model_name='logistic_regression'):
        """Make prediction with specified model"""
        if self.models is None:
            self.load_models()

        if self.models is None:
            return None

        model = self.models[model_name]

        # Scale features
        input_scaled = self.scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]

        return {
            'prediction': int(prediction),
            'probability': float(proba[1]),
            'probability_no_default': float(proba[0]),
            'confidence': float(max(proba))
        }

    def predict_all_models(self, input_data):
        """Get predictions from all models"""
        if self.models is None:
            self.load_models()

        if self.models is None:
            return None

        results = {}
        for model_name in self.models.keys():
            results[model_name] = self.predict(input_data, model_name)

        return results

    def get_model_metrics(self):
        """Return stored model performance metrics"""
        return {
            'logistic_regression': {
                'name': 'Logistic Regression',
                'accuracy': 0.63,
                'recall': 0.63,
                'precision': 0.26,
                'f1_score': 0.37,
                'roc_auc': 0.67
            },
            'random_forest': {
                'name': 'Random Forest',
                'accuracy': 0.68,
                'recall': 0.50,
                'precision': 0.27,
                'f1_score': 0.35,
                'roc_auc': 0.66
            },
            'xgboost': {
                'name': 'XGBoost',
                'accuracy': 0.79,
                'recall': 0.16,
                'precision': 0.31,
                'f1_score': 0.21,
                'roc_auc': 0.65
            }
        }

def interpret_risk(probability):
    """Interpret risk level from probability"""
    if probability < 0.10:
        return {
            'level': 'Very Low Risk',
            'color': 'green',
            'recommendation': ' **Approve** - Strong candidate with minimal default risk'
        }
    elif probability < 0.20:
        return {
            'level': 'Low Risk',
            'color': 'blue',
            'recommendation': ' **Approve** - Below average default risk'
        }
    elif probability < 0.40:
        return {
            'level': 'Medium Risk',
            'color': 'orange',
            'recommendation': ' **Review** - Above average risk, consider terms adjustment'
        }
    elif probability < 0.60:
        return {
            'level': 'High Risk',
            'color': 'red',
            'recommendation': ' **Caution** - High default probability, stricter terms required'
        }
    else:
        return {
            'level': 'Very High Risk',
            'color': 'darkred',
            'recommendation': ' **Decline** - Unacceptable default risk'
        }
def interpret_risk(probability):
    """
    Interpret risk level from probability

    Args:
        probability: Float between 0 and 1

    Returns:
        dict with level, color, recommendation
    """
    if probability < 0.10:
        return {
            'level': 'Very Low Risk',
            'color': 'green',
            'recommendation': ' **Approve** - Strong candidate with minimal default risk'

        }
    elif probability < 0.20:
        return {
            'level': 'Low Risk',
            'color': 'blue',
            'recommendation': '**Approve** - Below average default risk'
        }
    elif probability < 0.40:
        return {
            'level': 'Medium Risk',
            'color': 'orange',
            'recommendation': ' **Review** - Above average risk, consider terms adjustment'
        }
    elif probability < 0.60:
        return {
            'level': 'High Risk',
            'color': 'red',
            'recommendation': ' **Caution** - High default probability, stricter terms required'
        }
    else:
        return {
            'level': 'Very High Risk',
            'color': 'darkred',
            'recommendation': ' **Decline** - Unacceptable default risk'
        }


def calculate_expected_loss(loan_amount, default_probability, recovery_rate=0.10):
    """
    Calculate expected loss from a loan

    Args:
        loan_amount: Loan amount in dollars
        default_probability: Probability of default (0-1)
        recovery_rate: Expected recovery if default occurs (default 10%)

    Returns:
        Expected loss in dollars
    """
    return loan_amount * default_probability * (1 - recovery_rate)


def calculate_business_metrics(loan_amount, interest_rate, default_probability,
                               recovery_rate=0.10, false_positive_cost=2000):
    """
    Calculate business metrics for loan decision

    Args:
        loan_amount: Loan amount
        interest_rate: Annual interest rate (as decimal, e.g., 0.12 for 12%)
        default_probability: Probability of default
        recovery_rate: Expected recovery rate if default
        false_positive_cost: Cost of rejecting a good customer

    Returns:
        dict with various business metrics
    """
    # Expected revenue if loan is approved and paid back
    expected_revenue = loan_amount * interest_rate

    # Expected loss if default occurs
    expected_loss = calculate_expected_loss(loan_amount, default_probability, recovery_rate)

    # Net expected value
    net_expected_value = (expected_revenue * (1 - default_probability)) - expected_loss

    # Break-even default probability
    break_even_prob = expected_revenue / (loan_amount * (1 - recovery_rate))

    return {
        'expected_revenue': expected_revenue,
        'expected_loss': expected_loss,
        'net_expected_value': net_expected_value,
        'break_even_probability': break_even_prob,
        'recommendation': 'Approve' if net_expected_value > 0 else 'Decline'
    }


def get_feature_impact(feature_value, feature_name, percentile_data=None):
    """
    Determine if a feature value is good, neutral, or bad for default risk

    Args:
        feature_value: Value of the feature
        feature_name: Name of the feature
        percentile_data: Optional dict with 25th, 50th, 75th percentiles

    Returns:
        dict with impact assessment
    """
    # Default assessments (you can customize these thresholds)
    assessments = {
        'fico_range_low': {
            'excellent': (740, float('inf'), 'Excellent credit score'),
            'good': (670, 740, 'Good credit score'),
            'fair': (0, 670, 'Fair credit score - higher risk')
        },
        'dti': {
            'low': (0, 15, 'Low debt burden'),
            'moderate': (15, 25, 'Moderate debt burden'),
            'high': (25, float('inf'), 'High debt burden - risk factor')
        },
        'revol_util': {
            'low': (0, 30, 'Low credit utilization - positive'),
            'moderate': (30, 70, 'Moderate credit utilization'),
            'high': (70, float('inf'), 'High credit utilization - risk factor')
        },
        'loan_to_income': {
            'low': (0, 0.20, 'Affordable loan size'),
            'moderate': (0.20, 0.35, 'Moderate loan burden'),
            'high': (0.35, float('inf'), 'Large loan relative to income - risk')
        },
        'int_rate': {
            'low': (0, 10, 'Low interest rate'),
            'moderate': (10, 15, 'Moderate interest rate'),
            'high': (15, float('inf'), 'High interest rate - risky borrower')
        }
    }

    if feature_name not in assessments:
        return {'impact': 'neutral', 'description': 'No assessment available'}

    ranges = assessments[feature_name]

    for level, (min_val, max_val, description) in ranges.items():
        if min_val <= feature_value < max_val:
            impact = 'positive' if level in ['excellent', 'good', 'low'] else 'negative' if level in ['high', 'fair'] else 'neutral'
            return {
                'impact': impact,
                'level': level,
                'description': description,
                'value': feature_value
            }

    return {'impact': 'neutral', 'description': 'Value out of typical range'}
