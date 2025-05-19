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
    page_title="Validation and Statistical Analysis App - v.0.03",
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
    <h1>Welcome to the Validation and Statistical Analysis App (v.0.03) <span class='wave'>👋</span></h1>
    """, 
    unsafe_allow_html=True
)

# --- Loading Spinner and Confirmation ---
with st.spinner("Loading the analysis tools... Please wait!"):
    st.success("Analysis tools loaded successfully! 🎉 " \
    "\n Let's get started with your validation analysis!")
    # st.balloons()

# --- About Section ---
with st.expander("📖 About this App", expanded=True):
    st.markdown(
        """
        ### 📖 About this App
        \n This app is designed to assist laboratory professionals in performing validation and verification analyses. It provides a user-friendly interface for various statistical tests, including linear regression, imprecision analysis, method comparison analysis and outlier testing. For more information on each module, please refer to the sidebar.
        \n Please note, some modules in this app are in development (e.g., Kruskal-Wallis in Statistical Tests). Please DO NOT use modules flagged as "in development" until the message has been cleared.
        """)
with st.expander("📂 Data Upload and Templates", expanded=False):
    st.markdown("""### 📂 Data Upload and Templates
    \n For each module, your data must be in CSV format. Ensure your data is structured according to the provided templates, available from the module in the sidebar. 
    \n It is important that column names (i.e., Date, Material, Analyser, Sample ID, Batch ID and Lot Number) are consistent with the templates to ensure proper analysis. 
    \n You may insert as many analyte names as required - however, avoid introducing spaces or special characters **(e.g., @ : ; ! , . # < >)** in the column names.
    \n This app is intended for educational and informational purposes only. Always consult with a qualified professional for laboratory analyses and interpretations.
    """)
with st.expander("🛠️ Development, Deployment, and Contact", expanded=False):
    st.markdown("""
    #### 🛠️ Development, Deployment, and Contact
    > This app was developed using Streamlit and is hosted on GitHub. The source code is available for review and contributions.
    ---
    """)

# --- Available Modules ---
with st.expander("📖 Which modules are available?", expanded=False):
    st.markdown(
        """
        This app provides a comprehensive platform for performing various validation and verification analyses in laboratory settings.  
        Use the sidebar to navigate between the available tools.

        ### 📊 Analysis Modules include:

        - **Imprecision Analysis**
            - 📊 Intra-Well Imprecision
            - 🧪 Intra-Batch Precision
            - ⚗️ Inter-Batch Precision
            - 🥼 Total Imprecision

        - **📈 Linearity**

        - **Limits**
            - 💧 Limit of Blank (LOB)
            - 💧 Limit of Detection (LOD)
            - 💧 Limit of Quantitation (LOQ)  

        - **Method Comparison**
            - 😵‍💫 ANOVA
            - 🥼 Bland-Altman Analysis
            - ⚖️ Deming Regression
            - 📊 Passing-Bablok Analysis
                
        - **Outliers**
            - 🔎 Grubbs` Test
            - 🔎 Tietjen-Moore Test

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
            &copy; 2025 Validation and Statistical Analysis App. All rights reserved.
        </p>
    </footer>
    """, 
    unsafe_allow_html=True
)
# --- End of Main Page ---