import streamlit as st
from datetime import datetime
from utils import apply_app_styling, show_footer

st.set_page_config(
    page_title="Validation and Statistical Analysis App v. 1.04",
    page_icon=":computer:",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styling()

# Global styles
st.markdown("""
<style>
    body, p {
        font-size: 14px;
        line-height: 1.6;
        color: #2C3E50;
    }
    h2 {
        font-weight: 700;
        font-size: 28px;
        color: #34495E;
        margin-bottom: 0.5rem;
    }
    a {
        color: #4A90E2;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    a:hover {
        color: #1c6dd0;
        text-decoration: underline;
    }
    .sidebar .css-1d391kg {
        padding-top: 1rem;
    }
    .card {
        background:#f7f9fc; 
        padding:20px; 
        border-radius:10px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    footer {
        border-top: 1px solid #ddd;
        padding-top: 12px;
        font-size: 12px;
        color: #666;
        text-align: center;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔍 Validation Processes")
st.sidebar.info("Select a validation process from the sidebar above.")

# Greeting logic
hour = datetime.now().hour
if hour < 12:
    greeting = "Good Morning! 🌞"
    bg_color = "#4A90E2"
elif 12 <= hour < 18:
    greeting = "Good Afternoon! 🌻"
    bg_color = "#1f77b4"
else:
    greeting = "Good Evening! 🌙"
    bg_color = "#0B3D91"

st.markdown(f"""
    <div style="
        position: fixed;
        top: 70px;
        right: 30px;
        background-color: {bg_color};
        color: white;
        padding: 14px 24px;
        font-size: 16px;
        font-weight: 500;
        border-radius: 10px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-align: center;
        max-width: 280px;
        z-index: 1000;
        animation: fadeOutGreeting 6s forwards;
        ">
        {greeting}
    </div>
    <style>
        @keyframes fadeOutGreeting {{
            0% {{ opacity: 1; }}
            80% {{ opacity: 1; }}
            100% {{ opacity: 0; visibility: hidden; }}
        }}
    </style>
""", unsafe_allow_html=True)

# Main Title
st.markdown(
    f"""
    <h2 style='text-align: center; color: #2C3E50;'>Welcome to the Validation and Statistical Analysis App <span class='wave'>👋</span></h2>
    """, 
    unsafe_allow_html=True
)

with st.spinner("Loading the analysis tools... Please wait!"):
    st.success("Analysis tools loaded successfully! Let's get started with your validation analysis! 🎉")

# About section in card style
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📖 About this App")
    st.markdown("""
    This app is currently operating version 1.04, which includes several improvements and bug fixes.
    
    Designed to assist laboratory professionals in performing validation and verification analyses with a user-friendly interface for various statistical tests including linear regression, imprecision analysis, method comparison analysis, and outlier testing.

    Some modules are in development (e.g., Anderson-Darling, Bartlett, Cochran, Kolmogorov-Smirnov, Kruskal-Wallis, and Levene's Test). **DO NOT use modules flagged as "in development" until the message has been cleared.**
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Data upload section
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📂 Data Upload and Templates")
    st.markdown("""
    For each module, your data must be in CSV format. Ensure your data follows the provided templates available in the sidebar.

    Column names and order (Date, Material, Analyser, Sample ID, Batch ID, Lot Number) must be consistent to ensure proper analysis.

    You may insert as many analyte names as needed—avoid spaces or special characters (**e.g., @ : ; ! , . # < >**) in column names or data.

    The app automatically detects columns and provides options for selection.

    If you encounter issues, please consult documentation or contact support.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Modules section
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("📖 Available Modules")
    st.markdown("""
    Use the sidebar to navigate the comprehensive platform offering validation and verification analyses for laboratory settings.

    ### Modules include:
    - **Imprecision Analysis**
      - Intra-Well Imprecision
      - Intra-Batch Precision
      - Inter-Batch Precision
      - Inter-Analyser Precision
    - **Linearity**
      - Linear Regression
      - Recovery
    - **Limits**
      - Limit of Blank (LOB)
      - Limit of Detection (LOD)
      - Limit of Quantitation (LOQ)
    - **Method Comparison**
      - One-Way ANOVA
      - Two-Way Crossed ANOVA
      - Two-Way Nested ANOVA
      - Bland-Altman Analysis
      - Deming Regression
      - Passing-Bablok Analysis
    - **Outliers**
      - Grubbs` Test - *in development*
      - Tietjen-Moore Test - *in development*
    - **More Statistical Tests**
      - Anderson-Darling Test - *in development*
      - Bartlett's Test - *in development*
      - Chi-Squared Test
      - Cochran's Test
      - CUSUM Test
      - F-test
      - Kolmogorov-Smirnov Test - *in development*
      - Kruskal-Wallis Test - *in development*
      - Levene's Test - *in development*
      - Mann-Whitney U Test
      - P-P Plots
      - Q-Q Plots
      - Shapiro-Wilk Test - *in development*
      - T-test
      - Total Allowable Error (TEa)
      - Z-test
    """)
    st.markdown("</div>", unsafe_allow_html=True)
show_footer()
