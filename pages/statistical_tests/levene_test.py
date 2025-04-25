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

    with st.expander("📄 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"], label_visibility="collapsed")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head())
        else:
            df = None
            st.info("Awaiting file upload...")
            uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file is not None:
        try:

            with st.form("selection_form"):
                group_col = st.selectbox("Select the column that defines subgroups (e.g., Run, Instrument, Batch)", df.columns)
                material_col = st.selectbox("Select the Material column (e.g., QC1–QC5)", df.columns)
                analyte_col = st.selectbox("Select the Analyte column", df.select_dtypes(include=[np.number]).columns)

                selected_material = st.selectbox("Select a specific material to analyze", sorted(df[material_col].dropna().unique()))
                submitted = st.form_submit_button("Run Levene’s Test")

            if submitted:
                filtered_df = df[df[material_col] == selected_material][[group_col, analyte_col]].dropna()

                if filtered_df[group_col].nunique() < 2:
                    st.warning("You need at least two groups to compare variances.")
                else:
                    try:
                        groups = [group[analyte_col].values for _, group in filtered_df.groupby(group_col)]
                        stat, p_value = levene(*groups)

                        st.markdown(f"---\n### 🔬 Analyte: **{analyte_col}** — Material: **{selected_material}**")
                        st.write(f"**Levene’s Test Statistic:** {stat:.4f}")
                        st.write(f"**p-value:** {p_value:.4f}")

                        if p_value < 0.05:
                            st.error("❌ Significant differences in variance detected between groups.")
                        else:
                            st.success("✅ No significant differences in variance detected between groups.")
                    except Exception as e:
                        st.error(f"Error performing Levene’s test: {e}")
        except Exception as e:
            st.error(f"⚠️ Error loading file: {e}")