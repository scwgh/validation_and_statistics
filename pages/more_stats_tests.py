import streamlit as st
import sys
# sys.path.insert(0, '/workspaces/validation_app')  # Adjust if needed

from pages.statistical_tests import (
    anderson_darling, bartlett_test, f_test, t_test, cusum, cochran,
    kolmogorov_smirnov, kruskal_wallis, levene_test,
    mann_whitney_u, p_p_plots, q_q_plots, shapiro_wilk, tea, z_test, chi_squared
)
from utils import apply_app_styling

apply_app_styling()

st.title("📊 More Statistical Tests")

# Initialize session state
if "selected_test" not in st.session_state:
    st.session_state.selected_test = "Home"

# Descriptive intro
with st.expander("📘 Why perform more statistical tests?", expanded=True):
    st.markdown("""
    This section provides additional statistical tests commonly used in laboratory validation and verification analysis.
    Use the buttons below to launch the test you're interested in.
    \n **Disclaimer**: This section is **in development** and some of the features below may be prone to error. The following buttons are NOT ACTIVE and are prone to error: Anderson-Darling, Bartlett, Cochran, Kolmogorov-Smirnov, Kruskal-Wallis and Levene's Test. 
    \n If you encounter any further issues, please report them on the GitHub repository.
    """)

with st.expander("📘 How do I know which statistical test to choose?", expanded=False):
    st.markdown("""
    Understanding which statistical test is most appropriate for your data is sometimes a bit of a maze. This module incorporates a number of different statistical tests which are suitable for verification, validation, linearity adn 
    - **Anderson-Darling Test** – Determine if a sample of data comes from a specific distribution. 
    - **Bartlett's Test** – Assesses the equality of variances across multiple groups. It is sensitive to departures from normality and should only be used when data is approximately normally distributed.
    - **Chi-Squared Test** – Tests the independence of categorical variables or compares observed vs expected frequencies. Commonly used with contingency tables to assess distribution fit.
    - **Cochran's Test** – Used to detect whether a single variance is significantly larger than others in a group. Useful for identifying outliers in precision studies.
    - **CUSUM Test** – A sequential analysis technique for detecting small shifts in process mean over time. Helpful in ongoing quality control to monitor consistency.
    - **F-test** – Compares the variances of two populations. Commonly used to determine if two methods have significantly different variability.
    - **Kolmogorov–Smirnov Test** – A non-parametric test that compares a sample distribution with a reference probability distribution (e.g., normal) or compares two sample distributions.
    - **Kruskal-Wallis Test** – A non-parametric alternative to one-way ANOVA for comparing medians of more than two independent groups. Useful when data does not meet ANOVA assumptions.
    - **Levene's Test** – Tests the equality of variances across groups. Less sensitive to non-normality than Bartlett's test, making it a robust option in many real-world datasets.
    - **Mann-Whitney U Test** – A non-parametric test for comparing differences between two independent groups when the data is not normally distributed. It compares median values rather than means.
    - **P-P Plots** – Graphical tools to assess if a dataset follows a specified distribution (e.g., normal). Points should fall along the reference line if the data is normally distributed.
    - **Q-Q Plots** – Graphical tools to assess if a dataset follows a specified distribution (e.g., normal). Points should fall along the reference line if the data is normally distributed.
    - **Shapiro-Wilk Test** – Tests whether a sample comes from a normally distributed population. Often used as a preliminary check before performing parametric tests.
    - **T-test** – Compares the means of two independent groups to determine if they are significantly different. Assumes normally distributed data with equal variances.
    - **Total Allowable Error** – Evaluates whether the total observed error in a method is within acceptable clinical or regulatory limits. Integrates both bias and imprecision.
    - **Z-test** – Tests whether a sample mean differs significantly from a known population mean. Suitable when population variance is known or sample size is large.
    """)

# Define button labels and their module mapping
test_names = [
    "Anderson-Darling Test", "Bartlett's Test", "Chi-Squared Test", "Cochran's Test", "CUSUM Test",
    "F-test", "Kolmogorov–Smirnov Test", "Kruskal-Wallis Test", "Levene's Test",
    "Mann-Whitney U Test", "P-P Plot", "Q-Q Plot", "Shapiro-Wilk Test", "T-test", "Total Allowable Error", "Z-test"
]

module_map = {
    "Anderson-Darling Test": anderson_darling,
    "Bartlett's Test": bartlett_test,
    "Chi-Squared Test": chi_squared,
    "Cochran's Test": cochran,
    "CUSUM Test": cusum,
    "F-test": f_test,
    "Kolmogorov–Smirnov Test": kolmogorov_smirnov,
    "Kruskal-Wallis Test": kruskal_wallis,
    "Levene's Test": levene_test,
    "Mann-Whitney U Test": mann_whitney_u,
    "P-P Plot": p_p_plots,
    "Q-Q Plot": q_q_plots,
    "Shapiro-Wilk Test": shapiro_wilk,
    "T-test": t_test,
    "Total Allowable Error": tea,
    "Z-test": z_test,
}

# Create rows of buttons (3 per row)
for i in range(0, len(test_names), 3):
    cols = st.columns(3)
    for col, test_name in zip(cols, test_names[i:i+3]):
        if col.button(test_name):
            st.session_state.selected_test = test_name

# Load selected test module
selected_test = st.session_state.selected_test
if selected_test and selected_test != "Home":
    module = module_map.get(selected_test)
    if module and hasattr(module, "run"):
        module.run()
