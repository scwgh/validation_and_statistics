import numpy as np
import os
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
from scipy import stats
from scipy.stats import linregress
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import levene, ttest_ind
from scipy.odr import ODR, Model, RealData
import streamlit as st 
from datetime import datetime
import time     
import warnings
import plotly.graph_objects as go
from utils import apply_app_styling

# Set up the page config
st.set_page_config(
    page_title="Validation Templates",
    page_icon="📂",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styling() 

# Main title
st.title("📂 Validation Templates")

# Introduction markdown
st.markdown(
    """
    This app provides a comprehensive platform for performing various validation and verification analyses in laboratory settings. 
    Each section allows you to upload your data using the templates, explore outputs and export results as needed.

    #### 📂 Templates and Data Upload
    - For each module, you can upload your data in CSV format. Ensure your data is structured correctly according to the provided templates.
    - 3 templates are provided:
        - Linearity and Calibration Curve data (including standard data)
        - Imprecision and Method Comparison data (including QC, EQA and Patient data)
        - Limit of Detection (LOD) data

    """
)

# Section: 📈 Linearity and Calibration Template
st.subheader("📈 Linearity and Calibration Template")
st.markdown("Use this template to assess linearity and generate calibration curves for analytes across concentration ranges.")

with open("vnv_standard_template.csv", "rb") as file:
    st.download_button(
        label="⬇ Download Linearity Template",
        data=file,
        file_name="vnv_standard_template.csv",
        mime="text/csv"
    )

# Section: 🎯 Imprecision and Method Comparison Template
st.subheader(" 🎯 Imprecision and Method Comparison Template")
st.markdown("This template includes QC data for imprecision, as well as patient or EQA data for comparing analytical methods.")

with open("vnv_data_template.csv", "rb") as file:
    st.download_button(
        label="⬇ Download Imprecision & Method Comparison Template",
        data=file,
        file_name="vnv_data_template.csv",
        mime="text/csv"
    )

# Section: 🧪 Limit of Detection (LOD) Template
st.subheader("🧪 Limit of Detection (LOD) Template")
st.markdown("Use this template to estimate the LOD from replicate measurements near the detection threshold.")

with open("vnv_lod_template.csv", "rb") as file:
    st.download_button(
        label="⬇ Download LOD Template",
        data=file,
        file_name="vnv_lod_template.csv",
        mime="text/csv"
    )
