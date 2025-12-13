"""Reusable visualization components"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import streamlit as st

class LoanVisualizer:
    """Create standard visualizations for loan analysis"""

    @staticmethod
    def risk_gauge(probability, avg_rate=0.17):
        """Create risk gauge chart"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Default Risk (%)", 'font': {'size': 24}},
            delta = {'reference': avg_rate * 100, 'suffix': '% vs avg'},
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
                    'value': avg_rate * 100
                }
            }
        ))

        fig.update_layout(
            paper_bgcolor = "white",
            font = {'color': "darkblue", 'family': "Arial"},
            height = 400
        )

        return fig

    @staticmethod
    def model_comparison_bar(results):
        """Create bar chart comparing model predictions"""
        models = []
        probabilities = []

        for model_name, result in results.items():
            models.append(model_name.replace('_', ' ').title())
            probabilities.append(result['probability'] * 100)

        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=probabilities,
                text=[f"{p:.1f}%" for p in probabilities],
                textposition='outside',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
            )
        ])

        fig.add_hline(
            y=17,
            line_dash="dash",
            line_color="red",
            annotation_text="Avg Default Rate (17%)",
            annotation_position="right"
        )

        fig.update_layout(
            title="Predicted Default Probability by Model",
            xaxis_title="Model",
            yaxis_title="Default Probability (%)",
            showlegend=False,
            height=400
        )

        return fig

    @staticmethod
    def confusion_matrix_heatmap(cm, labels=['Non-Default', 'Default']):
        """Create confusion matrix heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20},
            showscale=True
        ))

        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )

        return fig

    @staticmethod
    def feature_importance_chart(feature_names, importances, top_n=15):
        """Create feature importance bar chart"""
        # Get top N features
        indices = importances.argsort()[-top_n:][::-1]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        fig = go.Figure(data=[
            go.Bar(
                y=top_features,
                x=top_importances,
                orientation='h',
                marker_color='steelblue'
            )
        ])

        fig.update_layout(
            title=f"Top {top_n} Most Important Features",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )

        return fig

    @staticmethod
    def roc_curve_plot(fpr, tpr, auc_score):
        """Create ROC curve"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color='darkorange', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='navy', width=2, dash='dash')
        ))

        fig.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate (Recall)',
            showlegend=True,
            height=500
        )

        return fig
