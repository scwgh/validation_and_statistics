import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from utils import apply_app_styling

def run():
    # Page title
    st.title("🧪 Multi-Way ANOVA")

    with st.expander("📘 What is Multi-Way ANOVA?"):
        st.markdown(""" 
        Multi-way ANOVA is an extension of one-way ANOVA that allows for the analysis of multiple independent variables (factors) simultaneously. It is used to determine how two or more factors influence a dependent variable and whether there are any interaction effects between these factors.

        - **Null hypothesis (H₀)**: All factor levels and interactions have no effect on the dependent variable.
        - **Alternative hypothesis (H₁)**: At least one factor or interaction has an effect.

        In this app, multi-way ANOVA will allow you to examine the effects of multiple QC levels (Materials), analyzers, or other factors that may be in your data.
        """)

    with st.expander("📘 Instructions:"):
        st.markdown(""" 
        1. **Upload your CSV file**  
        - Click the “📤 Upload CSV File” button.
        - Ensure your file is in **CSV format** and contains one row per sample.
        - Required columns:
            - Material: Identifies the QC level (e.g., QC1, QC2, etc.)
            - Analyser: The instrument used (optional, but useful for filtering)
            - Sample ID: Unique identifier for each sample
            - One or more **analyte columns** with numeric values

        2. **Data Check**  
        - Once uploaded, the app shows a **preview of your data**.
        - The analysis will focus only on samples where the Material column starts with “QC”.
        - Missing analyte values will be automatically excluded from the analysis.

        3. **Select Analyte**  
        - From the dropdown menu, choose the analyte you want to assess.

        4. **View ANOVA Results**  
        - If at least two QC levels are present, the app performs a multi-way ANOVA.
        - The output includes:
            - 📊 An ANOVA summary table
            - 📈 A violin plot showing data distribution by QC level
            - ✅ Interpretation of the **p-value**

        5. **Interpret Your Results**  
        - A **p-value < 0.05** suggests a statistically significant difference between at least two factor levels or their interaction.
        - Use this to evaluate analytical stability or QC performance drift.
        """)

    # --- File Upload ---
    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Raw Data Preview")
        st.dataframe(df.head())

        # Filter to only include rows where Material starts with "QC"
        df_qc = df[df['Material'].astype(str).str.startswith("QC")].copy()

        if df_qc.empty:
            st.warning("No QC data found. Ensure 'Material' column contains values like 'QC1', 'QC2', etc.")
        elif 'Analyser' not in df.columns or 'Material' not in df.columns:
            st.error("The CSV must contain 'Material' and 'Analyser' columns.")
        else:
            # Dynamically select analyte columns
            analyte_columns = [
                col for col in df.columns
                if col not in ['Material', 'Analyser', 'Sample ID']
                and pd.api.types.is_numeric_dtype(df[col])
            ]
            selected_analyte = st.selectbox("🔎 Select Analyte to Compare Across QCs", analyte_columns)

            # Drop rows with missing analyte values
            subset = df_qc[['Material', 'Analyser', selected_analyte]].dropna()

            if subset['Material'].nunique() < 2:
                st.warning("Not enough QC levels to perform ANOVA.")
            else:
                try:
                    # Perform multi-way ANOVA with 'Material' and 'Analyser' as factors
                    model = ols(f'Q("{selected_analyte}") ~ C(Material) + C(Analyser) + C(Material):C(Analyser)', data=subset).fit()
                    anova_table = anova_lm(model, typ=2)
                    p_values = anova_table['PR(>F)']

                    # Display ANOVA table
                    st.subheader("📊 ANOVA Summary")
                    st.dataframe(anova_table.round(4))
                    
                    # Interpretation of results
                    for factor in ['C(Material)', 'C(Analyser)', 'C(Material):C(Analyser)']:
                        p_value = p_values[factor]
                        st.markdown(f"**P-value for {factor}:** `{p_value:.4f}`")
                        st.markdown(f"**Significant effect?** {'✅ Yes' if p_value < 0.05 else '❌ No'}")

                    # Violin plot
                    st.subheader("🎻 Violin Plot")
                    fig = px.violin(
                        subset,
                        x='Material',
                        y=selected_analyte,
                        box=True,
                        points='all',
                        color='Analyser',
                        title=f"Distribution of {selected_analyte} by QC Level and Analyser",
                        labels={'Material': 'QC Level', selected_analyte: 'Concentration', 'Analyser': 'Instrument'},
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
