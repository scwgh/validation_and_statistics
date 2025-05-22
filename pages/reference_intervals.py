import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from utils import apply_app_styling

# --- Page Setup ---
st.set_page_config(
    page_title="Reference Interval Analysis",
    page_icon="👨‍👩‍👧‍👦",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styling()

# --- Title and Info ---
st.title("👨‍👩‍👧‍👦 Reference Interval Analysis - _in development_")

with st.expander("📘 What is Reference Interval Analysis?", expanded=True):
    st.markdown("""
        A **reference interval** defines the range of values expected for a healthy population.
        Typically, this is the central 95% of the data (2.5th to 97.5th percentile).

        This tool supports both:
        - **Non-parametric estimation** (using percentiles)
        - **Parametric estimation** (assuming normally distributed data)
    """)

with st.expander("📘 Reference Interval Metrics Explained:", expanded=False):
    st.markdown("""
    - **Lower Limit**: The 2.5th percentile (or mean - 1.96 * SD)
    - **Upper Limit**: The 97.5th percentile (or mean + 1.96 * SD)
    - **N**: Number of valid observations used in the calculation
    """)

with st.expander("📘 Instructions:", expanded=False):
    st.markdown("""
    **To get started:**

    1. Upload your `.csv` file containing results from healthy individuals.
    2. Ensure the following columns are present:
        - `Date`, `Analyser`, `BatchID`, `Sample ID`, `Date of Birth`, `Gender`
        - Analyte values should appear after the `Gender` column.
    3. Choose the reference interval method (Non-parametric or Parametric).
    4. View and download your calculated reference intervals.
    """)

# --- Reference Interval Calculation ---
def reference_intervals():
    with st.expander("📤 Upload Data", expanded=True):
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Data Preview")
        st.dataframe(df.head())

        # Validate required columns
        required_cols = ['Date of Analysis', 'Batch ID', 'Sample ID', 'Date of Birth', 'Gender']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return

        # Extract analyte columns (assumed to start after 'Gender')
        start_idx = df.columns.get_loc('Gender') + 1
        analyte_cols = df.columns[start_idx:]
        if len(analyte_cols) == 0:
            st.warning("No analyte columns found.")
            return

        # Optional filters
        with st.expander("🔍 Optional Filters", expanded=False):
            # Calculate Age in years from Date of Birth and Date of Analysis
            def calculate_age(dob, doa):
                try:
                    dob = pd.to_datetime(dob, errors='coerce')
                    doa = pd.to_datetime(doa, errors='coerce')
                    if pd.isna(dob) or pd.isna(doa):
                        return np.nan
                    return (doa - dob).days // 365
                except Exception:
                    return np.nan

            df['Calculated Age'] = df.apply(lambda row: calculate_age(row['Date of Birth'], row['Date of Analysis']), axis=1)

            # Filter by Gender
            genders = df['Gender'].dropna().unique().tolist()
            selected_gender = st.multiselect("Filter by Gender", genders, default=genders)
            df = df[df['Gender'].isin(selected_gender)]

            # Filter by Age
            valid_ages = df['Calculated Age'].dropna()
            if valid_ages.empty:
                st.warning("⚠ No valid ages could be calculated from 'Date of Birth' and 'Date of Analysis'.")
            else:
                min_age = int(valid_ages.min())
                max_age = int(valid_ages.max())
                selected_age = st.slider(
                    "Filter by Age (in years)",
                    min_value=min_age,
                    max_value=max_age,
                    value=(min_age, max_age)
                )
                df = df[df['Calculated Age'].between(selected_age[0], selected_age[1])]

        # Choose calculation method
        method = st.radio("Choose method for reference interval calculation:", ["Non-parametric", "Parametric (Assumes Normality)"])

        results = []
        for analyte in analyte_cols:
            data = df[analyte].dropna()
            if len(data) < 20:
                st.warning(f"⚠ Not enough data for {analyte}. Minimum 20 values required.")
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

            # Visualization
            with st.expander("📊 Visualize Analyte Distributions", expanded=False):
                selected_analyte = st.selectbox("Select an analyte to view distribution:", analyte_cols)
                analyte_data = df[selected_analyte].dropna()
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=analyte_data,
                    nbinsx=30,
                    name="Data",
                    marker_color="orange",
                    opacity=0.7
                ))

                if method == "Parametric (Assumes Normality)":
                    mean = analyte_data.mean()
                    std = analyte_data.std()
                    x_range = np.linspace(analyte_data.min(), analyte_data.max(), 100)
                    y_norm = norm.pdf(x_range, mean, std) * len(analyte_data) * (x_range[1] - x_range[0])
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=y_norm,
                        name="Normal Distribution",
                        mode="lines",
                        line=dict(color="black", dash="dash")
                    ))

                fig.update_layout(
                    title=f"Distribution of {selected_analyte}",
                    xaxis_title="Value",
                    yaxis_title="Count",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

reference_intervals()
