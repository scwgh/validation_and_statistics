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
from utils import apply_app_styling, show_footer
import kaleido  

st.set_page_config(
    page_title="Validation and Statistical Analysis App - v. 1.04",
    page_icon=":computer:",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styling()

st.sidebar.header("🔍 Validation Processes")
st.sidebar.info("Select a validation process from the sidebar above.")


hour = datetime.now().hour
if hour < 12:
    greeting = "Good Morning! 🌞"
elif 12 <= hour < 18:
    greeting = "Good Afternoon! 🌻"
else:
    greeting = "Good Evening! 🌙"

# st.markdown(f"""
#     <div id="greeting" style="
#         position: fixed; top: 10%; right: 10%; 
#         background-color: #4C9ED9; color: white; padding: 12px 20px; font-size: 16px;
#         font-weight: bold; border-radius: 8px; z-index: 1000; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         text-align: center; width: auto; max-width: 250px;">
#         {greeting}
#     </div>
#     <style>
#         #greeting {{
#             animation: fadeOut 4s forwards;
#         }}
#         @keyframes fadeOut {{
#             0% {{
#                 opacity: 1;
#             }}
#             99% {{
#                 opacity: 1;
#             }}
#             100% {{
#                 opacity: 0;
#                 visibility: hidden;
#             }}
#         }}
#     </style>
# """, unsafe_allow_html=True)

st.markdown(
    f"""
    <h2 style='text-align: center; color: #2C3E50;'>Welcome to the Validation and Statistical Analysis App (v. 1.03) <span class='wave'>👋</span></h2>
    """, 
    unsafe_allow_html=True
)

with st.expander("📖 About this App", expanded=True):
    st.markdown(
        """
        ### 📖 About this App
        \n This app is designed to assist laboratory professionals in performing validation and verification analyses. It provides a user-friendly interface for various statistical tests, including linear regression, imprecision analysis, method comparison analysis and outlier testing. For more information on each module, please refer to the sidebar.
        \n Please note, some modules in this app are in development (e.g., Anderson-Darling, Bartlett, Cochran, Kolmogorov-Smirnov, Kruskal-Wallis and Levene's Test in Statistical Tests). 
        \n **DO NOT use modules flagged as "in development" until the message has been cleared.**
        """)
    with st.spinner("Loading the analysis tools... Please wait!"):
        st.success("Analysis tools loaded successfully! Let's get started with your validation analysis! 🎉")
        # st.balloons()

with st.expander("📂 Data Upload and Templates", expanded=False):
    st.markdown("""### 📂 Data Upload and Templates
    \n For each module, your data must be in CSV format. Ensure your data is structured according to the provided templates, available from the module in the sidebar. 
    \n It is important that column names and ordering (i.e., Date, Material, Analyser, Sample ID, Batch ID and Lot Number) are consistent with the templates to ensure proper analysis. 
    \n You may insert as many analyte names as required - however, avoid introducing spaces or special characters **(e.g., @ : ; ! , . # < >)** in the column names or within your data.
    \n The app will automatically detect the column names and provide options for selecting the appropriate columns for analysis.
    \n If you encounter any issues with the templates or data upload, please refer to the documentation or contact support.
    """)
with st.expander("🛠️ Development, Deployment, and Contact", expanded=False):
    st.markdown("""
    #### 🛠️ Development, Deployment, and Contact
    This app was developed using Streamlit and is hosted on GitHub. The source code is available for review and contributions.
                \n If you have any issues or questions regarding this app .
    """)

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

        - **Linearity**
            - 📈 Linear Regression
            - 📈 Recovery
            
        - **Limits**
            - 💧 Limit of Blank (LOB)
            - 💧 Limit of Detection (LOD)
            - 💧 Limit of Quantitation (LOQ)  

        - **Method Comparison**
            - 🥼 One-Way ANOVA
            - 🥼 Two-Way Crossed ANOVA
            - 🥼 Two-Way Nested ANOVA
            - 🥼 Bland-Altman Analysis
            - 🥼 Deming Regression
            - 🥼 Passing-Bablok Analysis
                
        - **Outliers**
            - 🔎 Grubbs` Test - **in development**
            - 🔎 Tietjen-Moore Test - **in development**

        - **More Statistical Tests**
            - 📈 Anderson-Darling Test - **in development**
            - ⚖️ Bartlett's Test - **in development**
            - 🧮 Chi-Squared Test
            - ⚙️ Cochran's Test
            - 📈 CUSUM Test
            - 🔎 F-test
            - 📊 Kolmogorov-Smirnov Test - **in development**
            - 📏 Kruskal-Wallis Test - **in development**
            - ⚖️ Levene's Test - **in development**
            - 🧪 Mann-Whitney U Test
            - 📈 P-P Plots
            - 📊 Q-Q Plots
            - 📏 Shapiro-Wilk Test - **in development**
            - 🔬 T-test
            - 🧪 Total Allowable Error (TEa)
            - 📏 Z-test        
        """
    )
show_footer()
