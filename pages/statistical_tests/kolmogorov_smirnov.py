import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import kstest, norm

def run():
    st.header("📈 Kolmogorov–Smirnov (K-S) Test")

    with st.expander("📘 What is the Kolmogorov–Smirnov Test?", expanded=False):
        st.write("""
            The **K-S Test** compares a sample with a reference distribution (e.g., normal) to check if they are statistically similar.

            - Commonly used to test **normality**.
            - The **null hypothesis (H₀)**: the data follows the specified distribution.
            - The **alternative hypothesis (H₁)**: the data does not follow the specified distribution.
            - A **p-value < 0.05** suggests a significant difference from the reference distribution.
        """)

    with st.expander("📘 Instructions"):
        st.markdown("""
            1. Upload a CSV file where:
               - `Material` is in column 4 (index 3).
               - Analyte columns start from column 6 (index 5) onward.
            2. The K-S test will be run separately for each analyte, across each `Material`.
        """)

    # --- Upload CSV ---
    with st.expander("📤 Upload Your CSV File", expanded=True):
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            material_col = df.columns[3]
            analyte_cols = df.columns[5:]
            df[material_col] = df[material_col].astype(str)

            st.subheader("📋 Raw Data Preview")
            st.dataframe(df.head())

            if st.button("Run K-S Test"):
                for analyte in analyte_cols:
                    st.markdown(f"### 🔬 Analyte: **{analyte}**")

                    for material, group_df in df.groupby(material_col):
                        data = group_df[analyte].dropna()

                        if len(data) < 5:
                            st.warning(f"Not enough data for {analyte} in Material {material} (n < 5).")
                            continue

                        # Standardize the data for testing against normal distribution
                        standardized = (data - data.mean()) / data.std(ddof=1)
                        stat, p_value = kstest(standardized, 'norm')

                        st.write(f"**Material:** {material}")
                        st.write(f"- K-S Statistic: `{stat:.4f}`")
                        st.write(f"- p-value: `{p_value:.4f}`")

                        if p_value < 0.05:
                            st.error("❌ Data significantly differs from a normal distribution.")
                        else:
                            st.success("✅ Data does not significantly differ from a normal distribution.")

        except Exception as e:
            st.error(f"⚠️ Error loading data: {e}")
