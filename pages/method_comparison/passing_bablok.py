import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import linregress
from utils import apply_app_styling

# === Utility Functions ===

def calculate_r2(x, y, slope, intercept):
    y_pred = slope * x + intercept
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def simple_linear_regression(x, y):
    slope, intercept, _, _, _ = linregress(x, y)
    return slope, intercept

def perform_analysis(df, analyte, analyzer_1, analyzer_2, units):
    if analyte not in df.columns:
        st.warning(f"⚠ {analyte} column not found in data.")
        return None, None, None

    filtered = df[df['Analyser'].isin([analyzer_1, analyzer_2])]

    values_1 = filtered[filtered['Analyser'] == analyzer_1][['Sample ID', analyte]].dropna()
    values_2 = filtered[filtered['Analyser'] == analyzer_2][['Sample ID', analyte]].dropna()

    merged = pd.merge(values_1, values_2, on='Sample ID', suffixes=('_1', '_2'))
    if merged.shape[0] < 2:
        return None, None, None

    x = pd.to_numeric(merged[f'{analyte}_1'], errors='coerce')
    y = pd.to_numeric(merged[f'{analyte}_2'], errors='coerce')
    sample_ids = merged['Sample ID']

    slope, intercept = simple_linear_regression(x, y)
    r2 = calculate_r2(x, y, slope, intercept)

    results = {
        "Analyte": analyte,
        "Analyzer 1": analyzer_1,
        "Analyzer 2": analyzer_2,
        "Slope": round(slope, 2),
        "Intercept": round(intercept, 2),
        "R²": round(r2, 4),
        "n": len(x)
    }

    fig = plot_regression_plotly(analyte, x, y, sample_ids, slope, intercept, r2, analyzer_1, analyzer_2, units)
    return results, fig, merged

def plot_regression_plotly(analyte, x_data, y_data, sample_ids, slope, intercept, r2, analyzer_1, analyzer_2, units):
    x_line = np.linspace(min(x_data), max(x_data), 100)
    y_line = slope * x_line + intercept
    y_err = 1.96 * np.std(y_data - (slope * x_data + intercept))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        marker=dict(color="mediumslateblue", size=8),
        text=sample_ids,
        hovertemplate='<b>Sample ID:</b> %{text}<br><b>X:</b> %{x:.2f}<br><b>Y:</b> %{y:.2f}<extra></extra>',
        name="Data Point"
    ))

    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        line=dict(color='crimson', width=2),
        name=f'Regression Line: y = {slope:.2f}x + {intercept:.2f}<br>R² = {r2:.4f}'
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([x_line, x_line[::-1]]),
        y=np.concatenate([y_line - y_err, (y_line + y_err)[::-1]]),
        fill='toself',
        fillcolor='rgba(220, 20, 60, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        name="95% CI"
    ))

    fig.update_layout(
        title=f"Passing-Bablok Regression for {analyte}",
        xaxis_title=f"{analyzer_1} ({units})",
        yaxis_title=f"{analyzer_2} ({units})",
        plot_bgcolor='white',
        showlegend=True,
        title_font=dict(size=16, family="Arial", color="black")
    )

    return fig

# === Streamlit App ===

apply_app_styling()

st.title("📊 Passing-Bablok Comparison")

def passing_bablok():
    with st.expander("📘 What is Passing Bablok Regression?"):
        st.markdown("""
        **Passing Bablok regression** is a **non-parametric method comparison technique** that is robust to outliers and does not assume a specific error distribution.
        """)

    with st.expander("📘 Instructions:"):
        st.markdown("""
        1. Upload a CSV file containing `Date`, `Test`, `Analyser`, `Material`, `Sample ID`, `Batch ID`, `Lot Number`, and analyte columns.
        2. Select the two analyzers you want to compare.
        3. View regression plots and statistics.
        """)

    with st.expander("📤 Upload CSV File", expanded=True):
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="uploader")

    if not uploaded_file:
        return

    df = pd.read_csv(uploaded_file)
    st.write("### 📋 Data Preview")
    st.dataframe(df.head())

    units = st.selectbox("Select Units for Analytes", ["μmol/L", "mmol/L", "mg/dL", "g/L", "ng/mL"], index=0)

    required_cols = ['Analyser', 'Material', 'Sample ID']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns: {', '.join(required_cols)}")
        return

    materials = sorted(df['Material'].dropna().unique().tolist())
    material_options = ['All'] + materials
    selected_material = st.selectbox("Select Material", material_options)

    if selected_material != "All":
        df = df[df['Material'] == selected_material]

    analyzers = sorted(df['Analyser'].dropna().unique())
    if len(analyzers) < 2:
        st.error("At least two analyzers are required.")
        return

    analyzer_1 = st.selectbox("Select Analyzer 1", analyzers)
    analyzer_2 = st.selectbox("Select Analyzer 2", [a for a in analyzers if a != analyzer_1])

    metadata_cols = ['Date', 'Test', 'Analyser', 'Material', 'Sample ID', 'Batch ID', 'Lot Number']
    analyte_cols = [col for col in df.columns if col not in metadata_cols]


    if st.button("Run Passing Bablok Regression"):
        results = []
        for analyte in analyte_cols:
            result, fig, merged = perform_analysis(df, analyte, analyzer_1, analyzer_2, units)
            if result:
                results.append(result)
                st.subheader(analyte)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander(f"📊 Data for {analyte}"):
                    merged["% Difference"] = (
                        (merged[f"{analyte}_2"] - merged[f"{analyte}_1"]) / merged[f"{analyte}_1"]
                    ) * 100
                    merged["% Difference"] = merged["% Difference"].round(2)

                    st.dataframe(
                        merged.rename(columns={
                            f"{analyte}_1": analyzer_1,
                            f"{analyte}_2": analyzer_2
                        })[["Sample ID", analyzer_1, analyzer_2, "% Difference"]]
                    )
            else:
                st.warning(f"⚠ Insufficient data for {analyte}. Skipping...")

        if results:
            result_df = pd.DataFrame(results)
            st.success("✅ Analysis complete!")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Results", csv, "passing_bablok_results.csv", "text/csv")
        else:
            st.warning("No analytes had sufficient data.")

def run():
    passing_bablok()
