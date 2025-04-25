import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from utils import apply_app_styling

def run():
    st.title("🧪 Multi-Way ANOVA")

    with st.expander("📘 What is Multi-Way ANOVA?"):
        st.markdown(""" 
        Multi-way ANOVA allows analysis of how multiple independent variables (factors) affect a dependent variable. This includes interactions between factors.

        - **Factors**: Material (QC level), Analyser, Analyte, and optionally LotNo
        - **Null Hypothesis**: No effect from any factor or interaction
        - **Alternative**: At least one factor or interaction has an effect
        """)

    with st.expander("📘 Instructions"):
        st.markdown(""" 
        1. Upload a CSV with:
            - Material
            - Analyser
            - Sample ID
            - One or more numeric analyte columns
            - (Optional) LotNo
        2. The app will reshape the data and perform multi-way ANOVA using all factors.
        """)

    with st.expander("📤 Upload Your CSV File", expanded=True):
        uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📋 Raw Data Preview")
        st.dataframe(df.head())

        # Keep rows where Material starts with "QC"
        df_qc = df[df['Material'].astype(str).str.startswith("QC")].copy()

        if df_qc.empty:
            st.warning("No QC data found. Ensure 'Material' column contains values like 'QC1', 'QC2', etc.")
            return

        required_columns = {'Material', 'Analyser', 'Sample ID'}
        if not required_columns.issubset(df_qc.columns):
            st.error("Missing one or more required columns: 'Material', 'Analyser', 'Sample ID'")
            return

        # Identify analyte columns
        exclude_cols = {'Material', 'Analyser', 'Sample ID', 'LotNo'}
        analyte_cols = [col for col in df_qc.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_qc[col])]

        if not analyte_cols:
            st.error("No numeric analyte columns found.")
            return

        # Melt data into long format
        id_vars = ['Material', 'Analyser', 'Sample ID']
        if 'LotNo' in df_qc.columns:
            id_vars.append('LotNo')
        df_long = df_qc.melt(id_vars=id_vars, value_vars=analyte_cols,
                             var_name='Analyte', value_name='Value').dropna()

        st.subheader("📊 Long Format Data")
        st.dataframe(df_long.head())

        # Ensure enough levels
        if df_long['Material'].nunique() < 2:
            st.warning("Not enough QC levels for ANOVA.")
            return

        # Build dynamic formula
        factors = ['C(Material)', 'C(Analyser)', 'C(Analyte)']
        if 'LotNo' in df_qc.columns:
            factors.append('C(LotNo)')

        interactions = [f"{a}:{b}" for i, a in enumerate(factors) for b in factors[i+1:]]
        formula = "Value ~ " + " + ".join(factors + interactions)

        try:
            model = ols(formula, data=df_long).fit()
            anova_table = anova_lm(model, typ=2)

            st.subheader("📈 ANOVA Summary Table")
            st.dataframe(anova_table.round(4))

            # Interpretation
            p_values = anova_table['PR(>F)']
            for factor in anova_table.index:
                p = p_values[factor]
                st.markdown(f"**{factor}** — p-value: `{p:.4f}` → {'✅ Significant' if p < 0.05 else '❌ Not Significant'}")

            # Violin plot
            st.subheader("🎻 Violin Plot")
            color_by = 'LotNo' if 'LotNo' in df_qc.columns else 'Analyser'
            fig = px.violin(
                df_long,
                x="Material",
                y="Value",
                color=color_by,
                box=True,
                points="all",
                facet_col="Analyte",
                category_orders={"Material": sorted(df_long["Material"].unique())},
                title="Distribution by QC Level and Analyte"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Download ANOVA table
            csv_buffer = BytesIO()
            anova_table.to_csv(csv_buffer)
            st.download_button(
                "⬇ Download ANOVA Table",
                data=csv_buffer.getvalue(),
                file_name="multiway_anova_results.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error during ANOVA: {e}")
