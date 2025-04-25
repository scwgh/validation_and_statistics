import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import bartlett

def run():
    st.header("📏 Bartlett’s Test for Equal Variances")

    with st.expander("📘 What is Bartlett’s Test?", expanded=False):
        st.write("""
            Bartlett’s test is used to assess if **multiple samples have equal variances**.
            
            It assumes the data is **normally distributed**, and is sensitive to deviations from normality.
            
            **Use case:** Useful for validating assumptions before running ANOVA.
        """)

    with st.expander("📘 Instructions"):
        st.markdown("""
            1. Upload a CSV file where:
               - `Material` is in column 4 (index 3)
               - Analyte values start from column 6 (index 5) onward.
            2. Bartlett’s test will be run **separately for each Material**, across analyte columns.
        """)

    with st.expander("📤 Upload Your CSV File", expanded=True):
        uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            material_col = df.columns[3]
            analyte_data = df.iloc[:, 5:]
            analyte_data = analyte_data.apply(pd.to_numeric, errors='coerce')
            df[material_col] = df[material_col].astype(str)

            st.subheader("📊 Raw Data Preview")
            st.dataframe(df.head())

            if st.button("Run Bartlett's Test"):
                for material, group_df in df.groupby(material_col):
                    st.markdown(f"### 🧪 Material: **{material}**")

                    analyte_subset = group_df.iloc[:, 5:]
                    analyte_subset = analyte_subset.apply(pd.to_numeric, errors='coerce')
                    analyte_subset_clean = analyte_subset.dropna()

                    if analyte_subset_clean.empty or analyte_subset_clean.shape[0] < 2:
                        st.warning(f"Not enough valid data for Material: {material}")
                        continue

                    try:
                        groups = [analyte_subset_clean[col].values for col in analyte_subset_clean.columns]
                        stat, p_value = bartlett(*groups)

                        st.write(f"**Bartlett Test Statistic:** {stat:.4f}")
                        st.write(f"**p-value:** {p_value:.4f}")

                        if p_value < 0.05:
                            st.error("❌ Variances are significantly different — heteroscedasticity detected.")
                        else:
                            st.success("✅ Variances are not significantly different — homoscedasticity assumed.")

                    except Exception as e:
                        st.error(f"{material}: Error performing Bartlett’s test — {e}")

        except Exception as e:
            st.error(f"⚠️ Error loading data: {e}")
