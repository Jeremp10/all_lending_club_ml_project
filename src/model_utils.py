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
