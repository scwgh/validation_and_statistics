import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
from datetime import datetime
from utils import apply_app_styling

# Set up the page config
st.set_page_config(
    page_title="Reference Interval Analysis",
    page_icon="👨‍👩‍👧‍👦",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styling()

# --- Page Setup ---
st.title("👨‍👩‍👧‍👦 Reference Interval Analysis - _in development_")

# --- Method Explanation ---
with st.expander("📘 What is Reference Interval Analysis?", expanded=True):
    st.markdown("""
        A reference interval defines the range of values expected for a healthy population.
        Typically, this is the central 95% of the data (2.5th to 97.5th percentile).
        This tool calculates non-parametric or parametric reference intervals using data from the provided template. 
        """)
# --- How --- 
with st.expander("📘 Reference Interval Metrics Explained:", expanded=False):
    st.markdown(""" """)

# --- Instructions ---
with st.expander("📘 Instructions:", expanded=False): 
    st.markdown("""
    This tool allows you to assess **intra-well, intra-batch and inter-batch imprecision** across different levels of control or patient materials.

    To get started:

    1. **Upload your CSV file** – it should contain repeated measurements for the 
    2. Make sure your file includes:
    3. Once uploaded, the app will:

    """)

def reference_intervals():
    with st.expander("📤 Upload Data", expanded=True):
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Data Preview")
        st.dataframe(df.head())

        # Check required columns
        required_cols = ['Date', 'Analyser', 'BatchID', 'Sample ID', 'Date of Birth', 'Gender']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing one or more required columns: {', '.join(required_cols)}")
            return

        # Extract analyte columns (assumes analytes start after 'Gender')
        start_idx = df.columns.get_loc('Gender') + 1
        analyte_cols = df.columns[start_idx:]

        if analyte_cols.empty:
            st.warning("No analyte columns found.")
            return

        method = st.radio("Choose method", ["Non-parametric", "Parametric (Assumes Normality)"])

        results = []
        for analyte in analyte_cols:
            data = df[analyte].dropna()
            if len(data) < 20:
                st.warning(f"Not enough data for {analyte}. Need at least 20 values.")
                continue

            if method == "Non-parametric":
                lower = np.percentile(data, 2.5)
                upper = np.percentile(data, 97.5)
            else:
                mean = np.mean(data)
                std = np.std(data, ddof=1)
                lower = mean - 1.96 * std
                upper = mean + 1.96 * std

            results.append({
                "Analyte": analyte,
                "Lower Limit": round(lower, 2),
                "Upper Limit": round(upper, 2),
                "N": len(data)
            })

        if results:
            st.subheader("📈 Reference Interval Results")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)

            st.download_button(
                label="⬇ Download Reference Intervals",
                data=results_df.to_csv(index=False),
                file_name="reference_intervals.csv",
                mime="text/csv"
            )
