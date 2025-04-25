import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import levene

def run():
    st.header("🧪 Levene’s Test for Equality of Variances")

    with st.expander("📘 What is Levene’s Test?", expanded=False):
        st.write("""
            Levene’s Test is used to check whether multiple groups have **equal variances**.  
            It's an important assumption for parametric tests like ANOVA.

            - Unlike Bartlett’s test, Levene’s test does **not assume normality**.
            - A **p-value < 0.05** suggests that **variances are significantly different** between groups.
        """)

    with st.expander("📘 Instructions"):
        st.markdown("""
            1. Upload a CSV file where:
               - `Material` is in column 4 (index 3).
               - Analyte columns start from column 6 (index 5) onward.
            2. The test will compare **each analyte’s variance across QC levels**.
        """)

    with st.expander("📤 Upload Your CSV File", expanded=True):
        uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            material_col = df.columns[3]
            analyte_cols = df.columns[5:]
            df[material_col] = df[material_col].astype(str)

            st.subheader("📋 Raw Data Preview")
            st.dataframe(df.head())

            if st.button("Run Levene’s Test"):
                for analyte in analyte_cols:
                    st.markdown(f"### 🔬 Analyte: **{analyte}**")

                    subset = df[[material_col, analyte]].dropna()

                    if subset.empty or subset[material_col].nunique() < 2:
                        st.warning(f"Not enough groups to compare for {analyte}.")
                        continue

                    try:
                        groups = [group[analyte].values for _, group in subset.groupby(material_col)]
                        stat, p_value = levene(*groups)

                        st.write(f"**Levene’s Test Statistic:** {stat:.4f}")
                        st.write(f"**p-value:** {p_value:.4f}")

                        if p_value < 0.05:
                            st.error("❌ Significant differences in variance detected.")
                        else:
                            st.success("✅ No significant differences in variance detected.")

                    except Exception as e:
                        st.error(f"{analyte}: Error performing Levene’s test — {e}")

        except Exception as e:
            st.error(f"⚠️ Error loading data: {e}")
