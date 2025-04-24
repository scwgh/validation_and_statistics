import streamlit as st
import pandas as pd
import numpy as np
from scipy.odr import ODR, RealData, Model
import plotly.graph_objects as go
from utils import apply_app_styling

def run():
    apply_app_styling()

    st.title("⚖️ Deming Regression Analysis")

    with st.expander("📘 What is Deming Regression?"):
        st.markdown("""
        **Deming regression** is used when **both X and Y variables have measurement error**, often in method comparison studies.
        - **Slope**: Proportional bias (1 = ideal)
        - **Intercept**: Constant bias (0 = ideal)
        - **R²**: Strength of the linear relationship
        """)

    with st.expander("📘 Instructions:"):
        st.markdown("""
        1. Upload a CSV with `Material`, `Analyser`, `Sample ID`, and analyte columns.
        2. Choose two analyzers for comparison.
        3. Choose a material type and a unit.
        4. Run Deming Regression to view plots and download results.
        """)

    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])
        df = pd.read_csv(uploaded_file) if uploaded_file else None

    if df is not None:
        analyzers = df['Analyser'].dropna().unique()
        if len(analyzers) < 2:
            st.error("❌ Need at least two analyzers.")
        else:
            analyzer_1 = st.selectbox("Select Analyzer 1", analyzers, index=0)
            analyzer_2 = st.selectbox("Select Analyzer 2", analyzers, index=1)

            if analyzer_1 == analyzer_2:
                st.warning("⚠ Please select two different analyzers.")
            else:
                valid_materials = ["EQA", "Patient"]
                material_options = df['Material'].dropna().unique()
                filtered_materials = [m for m in material_options if m in valid_materials]
                selected_material = st.selectbox(
                    "Select Material Type",
                    options=filtered_materials,
                    index=0 if "EQA" in filtered_materials else 0
                )
                units = st.selectbox(
                    "Select Units for Analytes",
                    options=["μmol/L", "mmol/L", "mg/dL", "g/L", "ng/mL"],
                    index=0
                )
                deming_regression_analysis(df, analyzer_1, analyzer_2, selected_material, units)

def deming_regression_analysis(df, analyzer_1, analyzer_2, selected_material, units):
    ignore_cols = ['Material', 'Analyser', 'Date', 'Sample ID']
    analytes = [col for col in df.columns if col not in ignore_cols]
    if not analytes:
        st.error("❌ No analyte columns found.")
        return

    results_list = []

    for analyte in analytes:
        sub_df = df[(df['Material'] == selected_material) & df['Analyser'].isin([analyzer_1, analyzer_2])].copy()
        if sub_df.empty:
            continue

        sub_df[analyte] = pd.to_numeric(sub_df[analyte], errors='coerce')
        pivot = sub_df.pivot_table(index='Sample ID', columns='Analyser', values=analyte, aggfunc='mean')

        if analyzer_1 not in pivot or analyzer_2 not in pivot:
            continue

        pivot = pivot.dropna(subset=[analyzer_1, analyzer_2])
        if len(pivot) < 2:
            st.warning(f"⚠ Not enough data for {analyte} ({selected_material}). Skipping...")
            continue

        x, y = pivot[analyzer_1].values, pivot[analyzer_2].values

        def linear(B, x): return B[0] * x + B[1]
        model = Model(linear)
        odr_data = RealData(x, y)
        odr = ODR(odr_data, model, beta0=[1, 0])
        output = odr.run()

        slope, intercept = output.beta
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        results_list.append({
            'Analyte': analyte,
            'Material': selected_material,
            'Analyzer 1': analyzer_1,
            'Analyzer 2': analyzer_2,
            'Slope': round(slope, 4),
            'Intercept': round(intercept, 4),
            'R²': round(r_squared, 4),
            'n': len(pivot)
        })

        x_line = np.linspace(min(x), max(x), 100)
        y_line = slope * x_line + intercept

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers',
            name='Samples',
            marker=dict(color='steelblue'),
            hovertemplate=f"{analyzer_1}: %{{x:.2f}} {units}<br>{analyzer_2}: %{{y:.2f}} {units}<extra></extra>"
        ))

        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name=f'y = {slope:.2f}x + {intercept:.2f}',
            line=dict(color='red')
        ))

        fig.update_layout(
            title=f"Deming Regression for {analyte} ({selected_material})",
            xaxis_title=f"{analyzer_1} ({units})",
            yaxis_title=f"{analyzer_2} ({units})",
            height=500,
            plot_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    if results_list:
        results_df = pd.DataFrame(results_list)
        with st.expander("📊 View Regression Results"):
            st.success("✅ Regression complete.")
            st.dataframe(results_df)

            st.download_button(
                label="⬇ Download Results (CSV)",
                data=results_df.to_csv(index=False).encode('utf-8'),
                file_name="deming_regression_results.csv",
                mime="text/csv"
            )
    else:
        st.warning("⚠ No valid data for regression.")
