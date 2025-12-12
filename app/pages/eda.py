import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="EDA - Loan Default Predictor", layout="wide")

# Title
st.title(" Exploratory Data Analysis")
st.markdown("Interactive visualization of loan data and default patterns")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/raw_data/cleaned_loan_data.csv')

    # Create binary default column if it doesn't exist
    if 'default' not in df.columns:
        df['default'] = df['loan_status'].apply(
            lambda x: 1 if x == 'Charged Off' else 0
        )

    return df

df = load_data()

# Sidebar filters
st.sidebar.markdown("##  Data Filters")
st.sidebar.markdown("Filter the dataset to explore specific segments")

# Grade filter
selected_grades = st.sidebar.multiselect(
    "Select Loan Grades",
    options=sorted(df['grade'].unique()),
    default=sorted(df['grade'].unique())
)

# Filter data
df_filtered = df[df['grade'].isin(selected_grades)]

st.sidebar.metric("Filtered Loans", f"{len(df_filtered):,}")
st.sidebar.metric("Default Rate", f"{df_filtered['default'].mean():.1%}")

# Main content
tab1, tab2, tab3, tab4 = st.tabs([" Overview", " Loan Characteristics", " Borrower Profile", " Default Analysis"])

with tab1:
    st.markdown("## Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Loans", f"{len(df_filtered):,}")
    with col2:
        st.metric("Defaults", f"{df_filtered['default'].sum():,}")
    with col3:
        st.metric("Default Rate", f"{df_filtered['default'].mean():.1%}")
    with col4:
        st.metric("Avg Loan Amount", f"${df_filtered['loan_amnt'].mean():,.0f}")

    st.markdown("---")

    # Target distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Loan Status Distribution")
        status_counts = df_filtered['loan_status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Loan Outcomes",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Default vs Non-Default")
        default_counts = df_filtered['default'].value_counts()
        fig = go.Figure(data=[
            go.Bar(
                x=['Non-Default', 'Default'],
                y=default_counts.values,
                marker_color=['#2ecc71', '#e74c3c'],
                text=default_counts.values,
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="Binary Classification Target",
            yaxis_title="Count",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("## Loan Characteristics")

    # Loan amount distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Loan Amount Distribution")
        fig = px.histogram(
            df_filtered,
            x='loan_amnt',
            nbins=50,
            title="Distribution of Loan Amounts",
            labels={'loan_amnt': 'Loan Amount ($)'},
            color_discrete_sequence=['#3498db']
        )
        fig.add_vline(
            x=df_filtered['loan_amnt'].median(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Median: ${df_filtered['loan_amnt'].median():,.0f}"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Interest Rate Distribution")
        fig = px.histogram(
            df_filtered,
            x='int_rate',
            nbins=50,
            title="Distribution of Interest Rates",
            labels={'int_rate': 'Interest Rate (%)'},
            color_discrete_sequence=['#e74c3c']
        )
        fig.add_vline(
            x=df_filtered['int_rate'].median(),
            line_dash="dash",
            line_color="green",
            annotation_text=f"Median: {df_filtered['int_rate'].median():.1f}%"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Grade distribution
    st.markdown("### Loan Grade Distribution")
    grade_counts = df_filtered['grade'].value_counts().sort_index()
    fig = px.bar(
        x=grade_counts.index,
        y=grade_counts.values,
        labels={'x': 'Loan Grade', 'y': 'Count'},
        title="Distribution by Risk Grade (A=Best, G=Worst)",
        color=grade_counts.values,
        color_continuous_scale='RdYlGn_r'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("## 👤 Borrower Profile")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Annual Income Distribution")
        # Remove extreme outliers for better visualization
        income_filtered = df_filtered[df_filtered['annual_inc'] < 500000]
        fig = px.box(
            income_filtered,
            y='annual_inc',
            title="Borrower Income (Outliers >$500K removed for clarity)",
            labels={'annual_inc': 'Annual Income ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Median Income", f"${df_filtered['annual_inc'].median():,.0f}")

    with col2:
        st.markdown("### FICO Score Distribution")
        fig = px.histogram(
            df_filtered,
            x='fico_range_low',
            nbins=40,
            title="Credit Score Distribution",
            labels={'fico_range_low': 'FICO Score'},
            color_discrete_sequence=['#9b59b6']
        )
        st.plotly_chart(fig, use_container_width=True)

        st.metric("Median FICO", f"{df_filtered['fico_range_low'].median():.0f}")

    # DTI and Credit Utilization
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Debt-to-Income Ratio")
        fig = px.histogram(
            df_filtered[df_filtered['dti'] < 100],
            x='dti',
            nbins=50,
            title="DTI Distribution (Excluding outliers >100%)",
            labels={'dti': 'Debt-to-Income Ratio (%)'},
            color_discrete_sequence=['#f39c12']
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Credit Utilization")
        fig = px.histogram(
            df_filtered,
            x='revol_util',
            nbins=50,
            title="Credit Utilization Distribution",
            labels={'revol_util': 'Credit Utilization (%)'},
            color_discrete_sequence=['#1abc9c']
        )
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("## Default Analysis")
    st.markdown("Compare characteristics of defaults vs non-defaults")

    # FICO by default status
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### FICO Score: Defaults vs Non-Defaults")
        fig = px.box(
            df_filtered,
            x='default',
            y='fico_range_low',
            color='default',
            title="Credit Scores by Default Status",
            labels={'default': 'Defaulted', 'fico_range_low': 'FICO Score'},
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        fig.update_xaxes(ticktext=['Non-Default', 'Default'], tickvals=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # Stats
        non_default_fico = df_filtered[df_filtered['default']==0]['fico_range_low'].median()
        default_fico = df_filtered[df_filtered['default']==1]['fico_range_low'].median()
        st.info(f" Median FICO: Non-Default = {non_default_fico:.0f} | Default = {default_fico:.0f} | Difference = {non_default_fico - default_fico:.0f} points")

    with col2:
        st.markdown("### Interest Rate: Defaults vs Non-Defaults")
        fig = px.box(
            df_filtered,
            x='default',
            y='int_rate',
            color='default',
            title="Interest Rates by Default Status",
            labels={'default': 'Defaulted', 'int_rate': 'Interest Rate (%)'},
            color_discrete_map={0: '#2ecc71', 1: '#e74c3c'}
        )
        fig.update_xaxes(ticktext=['Non-Default', 'Default'], tickvals=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # Stats
        non_default_rate = df_filtered[df_filtered['default']==0]['int_rate'].median()
        default_rate = df_filtered[df_filtered['default']==1]['int_rate'].median()
        st.warning(f" Median Rate: Non-Default = {non_default_rate:.1f}% | Default = {default_rate:.1f}% | Difference = {default_rate - non_default_rate:.1f}%")

    # Default rate by grade
    st.markdown("### Default Rate by Loan Grade")
    default_by_grade = df_filtered.groupby('grade')['default'].agg(['sum', 'count', 'mean']).reset_index()
    default_by_grade.columns = ['Grade', 'Defaults', 'Total', 'Default Rate']
    default_by_grade['Default Rate'] = default_by_grade['Default Rate'] * 100

    fig = px.bar(
        default_by_grade,
        x='Grade',
        y='Default Rate',
        title="Default Rate Increases with Risk Grade",
        labels={'Default Rate': 'Default Rate (%)'},
        text='Default Rate',
        color='Default Rate',
        color_continuous_scale='Reds'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    st.success(" Key Insight: Higher-risk grades (E, F, G) have significantly higher default rates, validating Lending Club's risk classification.")

# Data table at bottom
with st.expander(" View Raw Data Sample"):
    st.dataframe(df_filtered.head(100), use_container_width=True)
    st.download_button(
        " Download Filtered Data",
        df_filtered.to_csv(index=False).encode('utf-8'),
        "filtered_loan_data.csv",
        "text/csv"
    )
