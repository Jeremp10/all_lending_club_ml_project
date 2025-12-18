"""Data loading and processing utilities"""

#import section
import pandas as pd
import streamlit as st
from pathlib import Path


class DataLoader:
    """Handles all data loading processes."""

    def __init__(self, base_dir='data'):
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / 'processed'
        self.raw_dir = self.base_dir / 'raw_data'


    @st.cache_data
    def load_clean_data(_self):
        """
        Cleaned data loaded with the default column.
        File: data/raw_data/cleaned_loan_data.csv
        """
        file_path = _self.raw_dir / 'cleaned_loan_data.csv'
        try:
            df = pd.read_csv(file_path)

            if 'default' not in df.columns:
                df['default'] = df['loan_status'].apply(
                    lambda x: 1 if x == 'Charged Off' else 0
                )
            return df

        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
            st.info("Please ensure cleaned_loan_data.csv exists in data/raw_data/")
            return None

    @st.cache_data
    def load_encoded_data(_self):
        """
        Load encoded data for modeling.
        File: data/processed/encoded_loan_data.csv
        """

        file_path = _self.processed_dir / 'encoded_loan_data.csv'

        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
            st.info("Please ensure encoded_loan_data.csv exists in data/processed/")
            return None

    @st.cache_data
    def load_raw_data(_self, nrows=None):
        """
        Loading raw data from kaggle
        """

        file_path = _self.raw_dir / 'accepted_2007_to_2018Q4.csv'

        try:
            if nrows:
                return pd.read_csv(file_path, nrows=nrows, low_memory=False)
            else:
                return pd.read_csv(file_path, low_memory=False)
        except FileNotFoundError:
            st.error(f"File not found: {file_path}")
            return None

    def get_feature_statistics(self,df):
        """
        Get summary statistics for the dataset
        Aim to return a dict with the key metrics.
        """
        if df is None:
            return None

        stats ={
            'total_loans': len(df),
            'columns': df.columns.tolist(),
            'default_rate': df['default'].mean() if 'default' in df.columns else None,
            'avg_loan_amount': df['loan_amnt'].mean(),
            'median_loan_amount': df['loan_amnt'].median(),
            'avg_income': df['annual_inc'].mean(),
            'median_income': df['annual_inc'].median(),
            'avg_fico': df['fico_range_low'].mean(),
            'avg_interest_rate': df['int_rate'].mean(),
            'avg_dti': df['dti'].mean(),
            'avg_credit_util': df['revol_util'].mean()
        }


        #Grade Distribution
        if 'grade' in df.columns:
            stats['grade_distribution'] = df['grade'].value_counts().to_dict()

        #loan_status_dist
        if 'loan_status' in df.columns:
            stats['loan_status_distribution'] = df['loan_status'].value_counts().to_dict()

        return stats

    def filter_by_grade(self, df, grades):
        """
        Filter dataframe by loan grades

        Arguments:
        df:Dataframe with grade column
        grades: List of grades to keep (e.g A,B,C)
        """


        if 'grade' not in df.columns:
            st.warning("'grade' column not found in df")
            return df

        return df[df['grade'].isin(grades)].copy()

    def get_available_files(self):
        """
        Available data files
        """

        files = {
            'raw': list(self.raw_dir.glob('*.csv')),
            'processed': list(self.processed_dir.glob('*.csv'))
        }
        return files


    @staticmethod
    def validate_columns(df, required_columns):
        """
        Validate that required columns exist in dataframe

        Arguments:
        df: Dataframe to check
        required_columns: List of required column names

        Returns:
        bool: True if all columns exist, False otherwise
        """

        missing = [col for col in required_columns if col not in df.columns]

        return len(missing) == 0, missing
