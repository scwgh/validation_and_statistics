import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from utils import apply_app_styling

def run():
    # Page title
    st.title("🧪 One-Way ANOVA")

    with st.expander("📘 What is One-Way ANOVA?"):
        st.markdown("""ANOVA, which stands for **Analysis of Variance**, is a statistical method used to determine whether there are any statistically significant differences between the means of three or more independent groups.

        In a **one-way ANOVA**, the analysis is performed using a single independent variable (or factor), such as QC level (e.g., QC1, QC2, QC3), and examines how this factor influences a continuous dependent variable (e.g., analyte concentration).

        The core idea is to compare the **variation between groups** to the **variation within groups**. If the variation between groups is significantly larger than the variation within groups, it suggests that the group means are not all the same.

        - **Null hypothesis (H₀)**: All group means are equal.
        - **Alternative hypothesis (H₁)**: At least one group mean is different.

        The result is quantified using a **p-value**. A p-value less than 0.05 typically indicates a statistically significant difference between groups.
        In the context of laboratory quality control, ANOVA helps determine if there is a **statistical difference in measured values across different QC materials**. 
        For example:
        - Is the concentration of a certain analyte measured consistently across QC1, QC2, and QC3?
        - Are results from different runs or batches statistically distinguishable?

        Using ANOVA allows you to:
        - Detect systematic shifts or drifts in QC levels.
        - Verify consistency and stability in analytical performance.
        - Support method validation or troubleshooting by quantifying variability.

        This app simplifies the process by allowing you to upload your results, select an analyte, and instantly see whether there's a significant difference in concentration between QC levels.
    """)

    with st.expander("📘 Instructions:"):
        st.markdown("""
        1. **Upload your CSV file**  
        - Click the “📤 Upload CSV File” button.
        - Make sure your file is in **CSV format** and contains one row per sample.
        - Required columns:
            - `Material`: Identifies the QC level (e.g., QC1, QC2, etc.)
            - `Analyser`: The instrument used (optional, but useful for filtering)
            - `Sample ID`: Unique identifier for each sample
            - One or more **analyte columns** with numeric values

        2. **Data Check**  
        - Once uploaded, the app shows a **preview of your data**.
        - The analysis will focus only on samples where the `Material` column starts with “QC” (e.g., QC1, QC2).
        - Missing analyte values will be automatically excluded from the analysis.

        3. **Select Analyte**  
        - From the dropdown menu, choose the analyte you want to assess.

        4. **View ANOVA Results**  
        - If at least two QC levels are present, the app performs a one-way ANOVA.
        - The output includes:
            - 📊 An ANOVA summary table
            - 📈 A violin plot showing data distribution by QC level
            - ✅ Interpretation of the **p-value**

        5. **Interpret Your Results**  
        - A **p-value < 0.05** suggests a statistically significant difference between at least two QC levels.
        - Use this to evaluate analytical stability or QC performance drift.

        --- 

        📝 *Tip:* For best results, ensure your dataset has sufficient QC replicates (at least 3 per level) to support reliable analysis.
        📝 *Tip:* To compare results between different lot numbers, make sure to include this detail within your data. If there is not data available in your dataframe, it will skip this step.
                    """)

    # --- File Upload ---
    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Raw Data Preview")
        st.dataframe(df.head())

        # Check if required columns exist
        required_columns = ['Material', 'Analyser', 'Sample ID']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            # Filter to only include rows where Material starts with "QC"
            df_qc = df[df['Material'].astype(str).str.startswith("QC")].copy()

            if df_qc.empty:
                st.warning("No QC data found. Ensure 'Material' column contains values like 'QC1', 'QC2', etc.")
            else:
                # Dynamically select analyte columns by excluding 'Material', 'Analyser', and 'Sample ID'
                analyte_columns = [
                    col for col in df.columns
                    if col not in ['Material', 'Analyser', 'Sample ID']
                    and pd.api.types.is_numeric_dtype(df[col])
                ]
                selected_analyte = st.selectbox("🔎 Select Analyte to Compare Across QCs", analyte_columns)

                # Drop rows with missing analyte values
                subset = df_qc[['Material', selected_analyte]].dropna()

                if subset['Material'].nunique() < 2:
                    st.warning("Not enough QC levels to perform ANOVA.")
                else:
                    try:
                        # Perform One-Way ANOVA
                        model = ols(f'Q("{selected_analyte}") ~ C(Material)', data=subset).fit()
                        anova_table = anova_lm(model, typ=2)
                        p_value = anova_table.loc['C(Material)', 'PR(>F)']

                        # Display ANOVA table
                        st.subheader("📊 ANOVA Summary")
                        st.dataframe(anova_table.round(4))
                        st.markdown(f"**P-value for QC comparison:** `{p_value:.4f}`")
                        st.markdown(f"**Significant difference?** {'✅ Yes' if p_value < 0.05 else '❌ No'}")

                        # Violin plot
                        st.subheader("🎻 Violin Plot")
                        fig = px.violin(
                            subset,
                            x='Material',
                            y=selected_analyte,
                            box=True,
                            points='all',
                            color='Material',  # Ensure color mapping works even with many QC levels
                            title=f"Distribution of {selected_analyte} by QC Level",
                            labels={'Material': 'QC Level', selected_analyte: 'Concentration'},
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Download ANOVA table
                        csv_buffer = BytesIO()
                        anova_table.to_csv(csv_buffer)
                        st.download_button(
                            "⬇ Download ANOVA Table",
                            data=csv_buffer.getvalue(),
                            file_name=f"anova_{selected_analyte}_results.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error(f"Error performing ANOVA: {e}")
