import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
from datetime import datetime
from utils import apply_app_styling

# Set up the page config
st.set_page_config(
    page_title="Imprecision Analysis",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styling()

# --- Page Setup ---
st.title("📊 Imprecision Analysis")

# --- Method Explanation ---
with st.expander("📘 What is Imprecision Analysis?", expanded=True):
    st.markdown("""
    Imprecision analysis is used to quantify how **consistent** a method is when measuring the same material multiple times.

    *****What types of imprecision do we assess?*****
    - **Intra-well imprecision**: Variation in repeated measurements within a single well or sample.
    - **Intra-batch imprecision**: Variation in repeated measurements within a single analytical run.
    - **Inter-batch imprecision**: Variation in measurements of the same sample across different runs or days.

    *****Why do we perform imprecision analysis?*****
    - Verifying analytical precision for method validation.
    - Assessing consistency of quality control materials.
    - Comparing instrument or assay performance.

    > 💡 Aim for %CVs within your lab's acceptable performance limits (e.g., <5% or <10% depending on the analyte).
    """)

# --- Instructions ---
with st.expander("📘 Instructions:"): 
    st.markdown("""
    This tool allows you to assess **intra-well, intra-batch and inter-batch imprecision** across different levels of control or patient materials.

    To get started:

    1. **Upload your CSV file** – it should contain repeated measurements for the same sample/material across different runs or days.
    2. Make sure your file includes:
    - `Material` (e.g., Control, Patient)
    - `QC Level` (e.g., QC1, QC2),
    - `Analyser` (e.g., Analyser1, Analyser2)
    - `Run` or `Day` - including in either short or long date format
    - One or more **analyte columns**. Please ensure your analyte names are consistent across the file.
    3. Once uploaded, the app will:
    - Group data by `Material`, `QC Level`, and `Run`
    - Calculate intra-batch, inter-batch, and total imprecision (CV%)
        - Intra-well imprecision will also be calculated if provided.
    - Output summaries and visualizations for each analyte

    ##### ℹ️ Results are reported in terms of **%CV (Coefficient of Variation)**, which reflects variability relative to the mean.

    """)

# # Output directory for saving plots
# output_dir = "Imprecision"
# os.makedirs(output_dir, exist_ok=True)

def precision_studies(df, selected_analyte):
    results, differences = [], []
    qc_df = df[df['Material'].str.startswith('QC', na=False)]

    # Inter-Batch plots
    inter_batch_groups = qc_df[qc_df['Test'] == 'Inter_Batch_Imprecision'].groupby(['Material', 'Analyser'])
    subplot_titles = [f"{material} - {analyzer}" for (material, analyzer) in inter_batch_groups.groups.keys()]
    num_plots = len(subplot_titles)

    fig = make_subplots(
        rows=(num_plots + 1) // 2,
        cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        horizontal_spacing=0.08,
        vertical_spacing=0.15
    )

    row, col = 1, 1

    for (material, analyzer), group in inter_batch_groups:
        group = group.copy()
        group['Date'] = pd.to_datetime(group['Date'], errors='coerce', dayfirst=True)
        group = group.dropna(subset=['Date', selected_analyte])

        if group.empty or len(group) < 2:
            continue

        overall_mean = round(group[selected_analyte].mean(), 2)
        sd = round(group[selected_analyte].std(), 2)

        group = group.sort_values('Date')
        group['Moving Average'] = group[selected_analyte].rolling(window=7, min_periods=1).mean()

        fig.add_trace(go.Scatter(
            x=group['Date'], y=group[selected_analyte], mode='markers',
            marker=dict(color='darkblue', size=6, opacity=0.6),
            name='Sample', showlegend=False
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=group['Date'], y=group['Moving Average'], mode='lines',
            line=dict(color='orange', width=2, dash='dot'),
            name='7-day MA', showlegend=False
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=group['Date'], y=[overall_mean] * len(group),
            mode='lines', line=dict(color='red', dash='solid'),
            name='Mean', showlegend=False
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=group['Date'], y=[overall_mean + 2 * sd] * len(group),
            mode='lines', line=dict(color='green', dash='dash'),
            name='+2SD', showlegend=False
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=group['Date'], y=[overall_mean - 2 * sd] * len(group),
            mode='lines', line=dict(color='green', dash='dash'),
            name='-2SD', showlegend=False
        ), row=row, col=col)

        col = 2 if col == 1 else 1
        row += 1 if col == 1 else 0

    if fig.data:
        fig.update_layout(
            height=400 * ((num_plots + 1) // 2),
            title_text=f"{selected_analyte} - Inter-Batch Imprecision (All Analyzers)",
            showlegend=False,
            template='plotly_white',
            margin=dict(t=50, l=30, r=30, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No Inter-Batch data available to plot for the selected analyte.")

    # Summary statistics for all imprecision types
    analyzer_means = {}

    for analyte in df.columns[5:]:
        for (material, analyzer, test), group in qc_df.groupby(['Material', 'Analyser', 'Test']):
            group = group.copy()
            group['Date'] = pd.to_datetime(group['Date'], errors='coerce', dayfirst=True)
            group = group.dropna(subset=['Date', analyte])

            if group.empty or len(group) < 2:
                continue

            overall_mean = round(group[analyte].mean(), 2)
            sd = round(group[analyte].std(), 2)
            nobs = group[analyte].count()
            cv = round((sd / overall_mean) * 100, 2) if overall_mean != 0 else np.nan
            sem = round(sd / np.sqrt(nobs), 2) if nobs != 0 else np.nan

            results.append({
                'Test': test,
                'Analyte': analyte,
                'n': nobs,
                'Material': material,
                'Analyser': analyzer,
                'Mean': overall_mean,
                'SD': sd,
                'CV (%)': cv,
                'SEM': sem
            })

            if test == "Inter_Batch_Imprecision":
                key = (analyte, material)
                analyzer_means.setdefault(key, {})[analyzer] = overall_mean

    # Inter-analyser differences
    analyser_comparison = []
    for (analyte, material), means_dict in analyzer_means.items():
        analyzers = list(means_dict.keys())
        if len(analyzers) < 2:
            continue

        mean_values = [means_dict[a] for a in analyzers]
        ia_mean = np.mean(mean_values)
        ia_sd = np.std(mean_values, ddof=1)
        ia_cv = (ia_sd / ia_mean) * 100 if ia_mean != 0 else np.nan

        analyser_comparison.append({
            'Analyte': analyte,
            'Material': material,
            'Inter-Analyser Mean': round(ia_mean, 2),
            'Inter-Analyser SD': round(ia_sd, 2),
            'Inter-Analyser CV (%)': round(ia_cv, 2)
        })

    differences = []
    for (analyte, material), means_dict in analyzer_means.items():
        analyzers = list(means_dict.keys())
        diffs = []

        for a1, a2 in combinations(analyzers, 2):
            m1, m2 = means_dict[a1], means_dict[a2]
            if m1 and m2:
                pct_diff = abs(m1 - m2) / ((m1 + m2) / 2) * 100
                diffs.append(pct_diff)

        if diffs:
            differences.append({
                'Analyte': analyte,
                'Material': material,
                'Mean % Difference Between Analysers': round(np.mean(diffs), 2)
            })

    stats_df = pd.DataFrame(results)
    diff_df = pd.DataFrame(differences)

    return (
        stats_df[stats_df['Test'] == 'Intra_Well_Imprecision'],
        stats_df[stats_df['Test'] == 'Intra_Batch_Imprecision'],
        stats_df[stats_df['Test'] == 'Inter_Batch_Imprecision'],
        diff_df,
        analyser_comparison
    )

# --- File Upload (Wrapped in Expander) ---
with st.expander("📤 Upload Your CSV File", expanded=True):
    st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
    uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    if len(df.columns) <= 5:
        st.warning("❗️ Not enough analyte columns detected.")
    else:
        analyte_options = df.columns[5:]
        selected_analyte = st.selectbox("🔎 Select Analyte to View", analyte_options)

        with st.spinner("Analyzing..."):
            intra_well_df, intra_batch_df, inter_batch_df, diff_df, analyser_comparison = precision_studies(df, selected_analyte)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # --- Results Output (Wrapped in Expander) ---
        with st.expander("📊 View and Download Results", expanded=True):
            st.subheader("📊 Summary Statistics")
            tab1, tab2, tab3 = st.tabs(["Intra-Well", "Intra-Batch", "Inter-Batch"])

            with tab1:
                st.dataframe(intra_well_df)
                st.download_button("⬇ Download Intra-Well", intra_well_df.to_csv(index=False), f"intra_well_results_{timestamp}.csv")

            with tab2:
                st.dataframe(intra_batch_df)
                st.download_button("⬇ Download Intra-Batch", intra_batch_df.to_csv(index=False), f"intra_batch_results_{timestamp}.csv")

            with tab3:
                st.dataframe(inter_batch_df)
                st.download_button("⬇ Download Inter-Batch", inter_batch_df.to_csv(index=False), f"inter_batch_results_{timestamp}.csv")

            st.subheader("📈 % Difference Summary")
            st.dataframe(diff_df)
            st.download_button("⬇ Download Differences", diff_df.to_csv(index=False), f"differences_{timestamp}.csv")

            st.subheader("📏 Inter-Analyser Summary Statistics")
            inter_analyser_df = pd.DataFrame(analyser_comparison)
            st.dataframe(inter_analyser_df)
            st.download_button("⬇ Download Inter-Analyser Stats", inter_analyser_df.to_csv(index=False), f"inter_analyser_stats_{timestamp}.csv")

