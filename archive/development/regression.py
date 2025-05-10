import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from utils import apply_app_styling
from typing import Optional

def get_linear():
    from archive.development.regression import linear
    return linear

def get_polynomial():
    from archive.development.regression import polynomial
    return polynomial


st.set_page_config(
    page_title="Regression",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styling()

st.title("📏 Regression")

# Store the uploaded file in session state
if 'df' not in st.session_state:
    st.session_state.df = None

# Store dropdown selections in session state
if 'x_axis' not in st.session_state:
    st.session_state.x_axis = None
if 'y_axis' not in st.session_state:
    st.session_state.y_axis = None
if 'identifier_column' not in st.session_state:
    st.session_state.identifier_column = None

with st.expander("📘 Regression: linear or polynomial?", expanded=True):
    st.markdown("""
    **Disclaimer:** this module is in development. DO NOT USE UNTIL FURTHER NOTICE.
    """)

with st.expander("📘 Instructions:"):
    st.markdown("""
    1. Upload a CSV file with your limit data.
    2. Ensure the file includes repeated low concentration samples labeled in the `Material` column (e.g., Standard 1).
                a. In the Sample ID column, label your dilutions appropriately. For example, Standard 1 - 1 in 10. Please refrain from using `:` notation.
    3. Select the analyte columns (numeric values expected).
    4. Click the button below to calculate LOD and LOQ. 
    """)

def upload_data() -> Optional[pd.DataFrame]:
    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"], label_visibility="collapsed")

        if uploaded_file:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.markdown("### 📖 Data Preview")
            st.dataframe(st.session_state.df.head())
        else:
            st.info("Awaiting file upload...")

# Add buttons for regression options
if st.session_state.df is not None:
    st.markdown("### Choose Regression Type")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Perform Linear Regression"):
            st.markdown("### 📈 Linear Regression Results")
            try:
                linear_results = linear(st.session_state.df)
                st.plotly_chart(linear_results['plot'], use_container_width=True)
                st.markdown(f"**R² Score:** {linear_results['r2_score']}")
            except Exception as e:
                st.error(f"An error occurred during linear regression: {e}")

    with col2:
        if st.button("Perform Polynomial Regression"):
            st.markdown("### 📉 Polynomial Regression Results")
            try:
                polynomial_results = polynomial(st.session_state.df)
                st.plotly_chart(polynomial_results['plot'], use_container_width=True)  # Fixing this typo
                st.markdown(f"**R² Score:** {polynomial_results['r2_score']}")
            except Exception as e:
                st.error(f"An error occurred during polynomial regression: {e}")
else:
    st.warning("Please upload a CSV file to proceed with regression analysis.")


# --- Optional Reference Section ---
with st.expander("📚 References"):
    st.markdown(""" """)
