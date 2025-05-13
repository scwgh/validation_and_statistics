import streamlit as st
import importlib
from utils import apply_app_styling

# Page setup
st.set_page_config(
    page_title="Method Comparison",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_app_styling()
st.title("⚖️ Method Comparison")

with st.expander("📘 Why perform Method Comparison?", expanded=True):
    st.markdown("""
    **Method comparison** involves comparing results from different methods or instruments to evaluate their agreement or consistency.
    This is critical in ensuring the reliability and performance of laboratory tests across different platforms, equipment, or methods.
    Here are the types of tests you can explore in this module:

    - **One-Way ANOVA**: Tests whether there are significant differences between the means of three or more independent groups (e.g., QC levels).
    - **Two-Way Crossed ANOVA**: Evaluates the effect of two independent factors (e.g., Material and Analyser) on a response variable, 
      assuming all combinations of these factors are measured. Useful for assessing interactions between factors.
    - **Two-Way Nested ANOVA**: Used when one factor (e.g., Material) is nested within another (e.g., Batch), assessing the variability within 
      and between these nested factors.
    - **Multi-Way ANOVA**: Extends One- and Two-Way ANOVA to analyze the effect of multiple factors (e.g., Material, Analyser, and Batch), 
      including their interactions, providing a more comprehensive analysis.
    - **Bland-Altman Analysis**: A graphical method to assess agreement between two measurement methods by plotting the difference vs. mean. 
      This helps identify any systematic bias or proportional differences.
    - **Deming Regression**: A linear regression method that accounts for measurement error in both variables, ideal for comparing two 
      quantitative methods, especially when neither method can be considered a true reference.
    - **Passing-Bablok Regression**: A non-parametric regression used to compare methods, robust against outliers and assumptions about data 
      distribution. It provides a regression line without assuming normal distribution of errors.

    These tools support analytical method validation, troubleshooting, and cross-platform harmonization.

    **Disclaimer:** The _ANOVA_ modules are in development. If you identify any errors using these modules, please log them on GitHub.
    """)


# Define available tests and their module names
test_options = [
    "One-Way ANOVA",
    "Two-Way Crossed ANOVA",
    "Two-Way Nested ANOVA",
    "Multi-Way ANOVA",    
    "Bland-Altmann Analysis",
    "Deming Regression",
    "Passing Bablok Regression",
]

module_map = {
    "One-Way ANOVA": "one_way_anova",
    "Two-Way Crossed ANOVA": "two_way_crossed",
    "Two-Way Nested ANOVA": "two_way_nested",
    "Multi-Way ANOVA": "multi_way_anova",
    "Bland-Altmann Analysis": "bland_altmann",
    "Deming Regression": "deming_regression",
    "Passing Bablok Regression": "passing_bablok",
}

# Create a state variable for selected test if not already initialized
if "selected_test" not in st.session_state:
    st.session_state.selected_test = None

# Layout: Button row for test selection
with st.expander("Select a test to get started..", expanded=True):
    st.markdown("Choose a test to perform from the options below:")
    cols = st.columns(len(test_options))
    for i, test in enumerate(test_options):
        if cols[i].button(test):
            st.session_state.selected_test = test

# Only run the module if a test has been selected
if st.session_state.selected_test:
    selected_test = st.session_state.selected_test
    if selected_test in module_map:
        selected_module = module_map[selected_test]
        module_path = f"pages.method_comparison.{selected_module}"

        try:
            module = importlib.import_module(module_path)
            if hasattr(module, "run"):
                module.run()
            else:
                st.error(f"The module `{selected_module}.py` must have a `run()` function.")
        except ModuleNotFoundError:
            st.error(f"Module `{module_path}` not found. Check the file structure.")
        
else:
    st.markdown("  ")  # Keeps space when no test is selected.
