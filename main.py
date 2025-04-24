# --- Import Required Libraries ---
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import seaborn as sns
import plotly.graph_objects as go
import streamlit as st

from scipy import stats
from scipy.stats import linregress, levene, ttest_ind
from scipy.odr import ODR, Model, RealData

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from utils import apply_app_styling
import kaleido  

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Validation Analysis App",
    page_icon=":computer:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Apply Custom Styling ---
apply_app_styling()

# --- Sidebar Navigation ---
st.sidebar.header("🔍 Validation Processes")
st.sidebar.success("Select a validation process from the sidebar above.")

# --- Main Page Welcome Message ---
st.markdown(
    """
    <h1>Welcome to the Validation and Verification Analysis App! <span class='wave'>👋</span></h1>
    """, 
    unsafe_allow_html=True
)

# --- Loading Spinner and Confirmation ---
with st.spinner("Loading the analysis tools... Please wait!"):
    st.success("Analysis tools loaded successfully! 🎉 Let's get started with your validation analysis!")
    st.balloons()

# --- About Section ---
st.markdown(
    """
    ### 📖 About this App
    > Each section allows you to upload your data using the templates (see below), explore outputs, and export results as needed.

    #### 📂 Templates and Data Upload
    > For each module, upload your data in CSV format. Ensure your data is structured according to the provided templates, available in the sidebar.

    #### 📜 Disclaimer
    > This app is intended for educational and informational purposes only. Always consult with a qualified professional for laboratory analyses and interpretations.
    
    #### 🛠️ Development, Deployment, and Contact
    > This app was developed using Streamlit and is hosted on GitHub. The source code is available for review and contributions.
    ---
    """
)

# --- Available Modules ---
with st.expander("📖 Which modules are available?", expanded=False):
    st.markdown(
        """
        This app provides a comprehensive platform for performing various validation and verification analyses in laboratory settings.  
        Use the sidebar to navigate between the available tools.

        ### 📊 Available Analysis Modules

        - **Imprecision Analysis**
            - 📊 Intra-Well Imprecision
            - 🧪 Intra-Batch Precision
            - ⚗️ Inter-Batch Precision
            - 🥼 Total Precision

        - **Standard Curve and Linearity**
            - 👨🏻‍🔬 Calibration Curve
            - 👩🏽‍🔬 Response Curve

        - 🧪 Limit of Detection (LOD)

        - **Method Comparison**
            - 🧪 ANOVA
            - 🥼 Bland-Altman Analysis
            - ⚖️ Deming Regression
            - 📊 Passing-Bablok Analysis

        - **More Statistical Tests**
            - 🧮 Chi-Squared Test
            - ⚙️ Cochran's Test
            - 📈 CUSUM Test
            - 🔎 F-test
            - 📊 Kolmogorov-Smirnov Test
            - 📏 Kruskal-Wallis Test
            - ⚖️ Levene's Test
            - 🧪 Mann-Whitney U Test
            - 📈 P-P Plots
            - 📊 Q-Q Plots
            - 📏 Shapiro-Wilk Test
            - 🔬 T-test
            - 🧪 Total Allowable Error (TEa)
            - 📏 Z-test
        """
    )


# --- Footer ---
st.markdown(
    """
    ---
    <footer>
        <p style='text-align: center;'>
            &copy; 2023 Validation and Verification Analysis App. All rights reserved.
        </p>
    </footer>
    """, 
    unsafe_allow_html=True
)
# --- End of Main Page ---