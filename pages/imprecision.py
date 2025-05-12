import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
from datetime import datetime
from utils import apply_app_styling, check_westgard_rules, grubbs_test

# Set up the page config
st.set_page_config(
    page_title="Imprecision Analysis",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styling()

# --- Page Setup ---
st.title("📊 Imprecision Analysis (with Westgard and Grubbs' tests)")

# --- Method Explanation ---
with st.expander("📘 What is Imprecision Analysis?", expanded=True):
    st.markdown("""
    Imprecision analysis is used to evaluate random error associated with a measurement. Measurement of laboratory analytical error falls into two main categories: \n "***systematic error***" and "***random error***". 
    \n **Systematic errors** are predictable problems influencing observations consistently in one direction, while **random errors** are more unpredictable. Systematic errors are typically demonstrated  by bias (for example, ongoing negative bias for a QC), while random errors by the imprecision measured by the coefficient of variation (CV, %) (for example, one measurement outside of the expected range with all other repeat measurements within range). 
    \n Imprecision affects the reproducibility and repeatability of results. Reproducibility is defined as the closeness of the results of successive measurements under changed conditions. Repeatability is the closeness of the results of at least twenty successive measurements under similar conditions. By contrast, bias is the average deviation from a true value with minimal contribution of imprecision while inaccuracy is the deviation of a single measurement from the true value with significant contribution by imprecision. Multiple measurements, at least twenty and preferably forty, are therefore required for calculating imprecision as well as bias.
                
    \n ****What types of imprecision are we interested in?****
    - **Intra-well imprecision**: Variation in repeated measurements within a single well or sample.
    - **Intra-batch imprecision**: Variation in repeated measurements within a single bathch.
    - **Inter-batch imprecision**: Variation in measurements of the same sample across different batches run across days.
                
    \n From this module, you can visualise and evaluate your data, applying Westgard rules and performing a Grubbs' test to identify statistically significant outliers. For more information on **Grubbs' Test**, please navigate to the `Outlier` page.
    """)
# --- How --- 
with st.expander("📘 Imprecision Metrics Explained:", expanded=False):
    st.markdown("### 📐 Imprecision Metrics Explained")
    st.markdown("**🔹 Standard Deviation (SD)**: A measure of the dispersion of individual measurements around the mean. A low standard deviation suggests values are close to the mean; while a high standard deviation indicates values are spread over a wider range. SD is often used to determine if a value is an outlier.")
    st.latex(r'''\text{SD} = \sqrt{ \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})^2 }''')

    st.markdown("**🔹 Coefficient of Variation (CV)**: The coefficient of variation (CV) is defined as a ratio of standard deviation to mean. CV should only be used for data which has a meaningful zero and therefore can be applied as a relative comparison between two measurements. It may also be defined as the standard deviation expressed as a percentage of the mean.")
    st.latex(r'''\text{CV}(\%) = \left( \frac{\text{SD}}{\bar{x}} \right) \times 100''')

    st.markdown("**🔹 Bias**: Bias is expressed as a percentage of the mean measured value versus the expected value.")
    st.latex(r'''\text{Bias (\%)} = \left( \frac{\bar{x} - \mu}{\mu} \right) \times 100''')

    st.markdown("**🔹 Grubb's test**, also known as Extreme Studentized Deviate (ESD) test, is used to identify outliers which are statistically significant from a univariate data set that follows an approximately normal distribution.")

# --- Instructions ---
with st.expander("📘 Instructions:", expanded=False): 
    st.markdown("""
    This tool allows you to assess **intra-well, intra-batch, and inter-batch imprecision** across different levels of control or patient materials.

    1. **Upload your CSV file** – it should contain repeated measurements for the same sample/material across different runs or days.

    2. Your file should include the following columns:
    - `Material` (e.g., QC1, QC2, QC3)
    - `Analyser` (e.g., Waters TQ-D 1, Waters TQ-D 2)
    - `Date` – can be in short or long date format
    - One or more **analyte columns** – ensure consistent naming and avoid use of special characters

    3. After upload, the app will:
    - Group data by `Material`, `QC Level`, and `Analyte`
    - Calculate intra-well, intra-batch, and inter-batch imprecision
    - Generate summary statistics and visualizations for each analyte
    - Allow export of calculated results and outlier summaries
    - Westgard rules are applied to imprecision charts.
    - Use the toggle in the sidebar to enable or disable rule overlays.

    ### 📊 Outlier Detection – Grubbs’ Test:
    - Grubbs’ Test identifies statistical outliers in your dataset (based on extreme deviation from the mean).
    - Use the dropdown checkbox to:
        - **Apply Grubbs’ Test** to identify potential outliers.
        - **Optionally exclude** flagged outliers from calculations.

    """)


with st.expander("ℹ️ What are the Westgard Rules?"):
    st.markdown("""
    The **Westgard Rules** are a set of statistical criteria used to monitor analytical performance and detect potential errors in quality control (QC) data. Each rule examines patterns or outliers based on the mean and standard deviation (SD) of control measurements. Here's what each rule means:

    \n ⚠️ **Warning Rule**
    \n- **1₂s**: One control result exceeds **±2 SD** from the mean. This is a **warning**, not a rejection.

    \n ❌ **Rejection Rules**
    \n - **1₃s**: One control result exceeds **±3 SD**. This suggests a **random error**.
    \n - **2₂s**: Two consecutive control results exceed **±2 SD in the same direction**. Indicates **systematic error**.
    \n - **R₄s**: One control result exceeds the mean by +2 SD and the next by -2 SD (or vice versa), giving a **range > 4 SD**. Detects **random error**.
    \n - **4₁s**: Four consecutive results exceed **±1 SD in the same direction**. Indicates **systematic shift**.
    \n - **10x**: Ten consecutive results are on the **same side of the mean**. Flags **bias** or drift.
    \n - **7T**: Seven points show a **consistent trend**, either increasing or decreasing. Suggests **progressive change**.
    \n - **8x**: Eight consecutive points lie on the **same side of the mean**. Another test for **systematic error**.

    \n Each rule helps identify potential issues in assay performance. You can toggle which rules are applied using the sidebar checkboxes.
    """)



def precision_studies(df, selected_analyte, rules_enabled, grubbs_outliers):
    results = []
    outlier_indices = []
    filtered_data = df.copy()  # default if not computed
    qc_df = df[df['Material'].str.startswith('QC', na=False)]

    inter_batch_groups = qc_df[qc_df['Test'] == 'Inter_Batch_Imprecision'].groupby(['Material', 'Analyser'])
    
    subplot_titles = [f"{material} - {analyzer}" for (material, analyzer) in inter_batch_groups.groups.keys()]
    num_plots = len(subplot_titles)

    if num_plots > 0:
        fig = make_subplots(
            rows=(num_plots + 1) // 2,
            cols=2,
            subplot_titles=subplot_titles,
            shared_xaxes=True,
            horizontal_spacing=0.08,
            vertical_spacing=0.15
        )
    else:
        st.info("No Inter-Batch data available to plot for the selected analyte.")

    row, col = 1, 1

    # --- Plot All Data Points with Error Bars (Including Westgard Alerts) ---
    for (material, analyzer), group in qc_df.groupby(['Material', 'Analyser']):
        group = group.copy()
        group['Date'] = pd.to_datetime(group['Date'], errors='coerce', dayfirst=True)
        group = group.dropna(subset=['Date', selected_analyte, 'Analyser', 'Material'])

        if group.empty or len(group) < 2:
            continue

        overall_mean = round(group[selected_analyte].mean(), 2)
        sd = round(group[selected_analyte].std(), 2)

        fig.add_trace(go.Scatter(
            x=group['Date'], y=group[selected_analyte], mode='markers',
            marker=dict(color='darkblue', size=6, opacity=0.6),
            name='Sample', showlegend=True
        ), row=row, col=col)

        # Add Error Bars (Mean ± SD)
        fig.add_trace(go.Scatter(
            x=group['Date'], y=[overall_mean] * len(group),
            mode='lines', line=dict(color='dodgerblue', dash='solid'),
            name='Mean', showlegend=True
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=group['Date'], y=[overall_mean + (2*sd)] * len(group),
            mode='lines', line=dict(color='green', dash='dash'),
            name='+2 SD', showlegend=True
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=group['Date'], y=[overall_mean - (2*sd)] * len(group),
            mode='lines', line=dict(color='green', dash='dash'),
            name='-2 SD', showlegend=True
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=group['Date'], y=[overall_mean + (3*sd)] * len(group),
            mode='lines', line=dict(color='red', dash='dash'),
            name='+3 SD', showlegend=True
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=group['Date'], y=[overall_mean - (3*sd)] * len(group),
            mode='lines', line=dict(color='red', dash='dash'),
            name='-3 SD', showlegend=True
        ), row=row, col=col)

        # --- Apply Westgard Alerts ---
        rule_alerts = check_westgard_rules(group[selected_analyte].tolist(), overall_mean, sd, rules_enabled)
        for i, rule in rule_alerts:
            fig.add_trace(go.Scatter(
                x=[group['Date'].iloc[i]], y=[group[selected_analyte].iloc[i]],
                mode='markers', marker=dict(color='crimson', size=9, symbol='x'),
                name=f'Violation: {rule}', showlegend=False
            ), row=row, col=col)

        # After applying Grubbs' test and identifying outliers, gather the details
        outlier_details = []
        if grubbs_outliers.get("perform_grubbs"):
            outlier_alerts = grubbs_test(group[selected_analyte])
            outlier_indices = outlier_alerts["Outlier Indices"]
            st.info(f"Grubbs' Test applied to {material} [{analyzer}]. No outliers identified.")

            for idx in outlier_indices:
                fig.add_trace(go.Scatter(
                    x=[group.loc[idx, 'Date']],
                    y=[group.loc[idx, selected_analyte]],
                    mode='markers',
                    marker=dict(color='darkorchid', size=10, symbol='square'),
                    name='Grubbs` Outlier',
                    showlegend=False
                ), row=row, col=col)

            if outlier_indices:
                outlier_details = [
                    f"{selected_analyte} = {group.loc[idx, selected_analyte]} ({group.loc[idx, 'Material']})"
                    for idx in outlier_indices
                ]
                st.error(f"Grubbs' Test applied to {material} [{analyzer}]. {len(outlier_indices)} outlier(s): {' | '.join(outlier_details)}")

                if grubbs_outliers.get("exclude_grubbs"):
                    group = group.drop(index=outlier_indices)

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
        st.info("No data available for plotting.")

    # -- Imprecision statistics for all --
    analyzer_means = {}

    for analyte in df.columns[7:]:
        for (material, analyzer, test), group in qc_df.groupby(['Material', 'Analyser', 'Test']):
            group = group.copy()
            group['Date'] = pd.to_datetime(group['Date'], errors='coerce', dayfirst=True)
            group = group.dropna(subset=['Date', analyte])

            if group.empty or len(group) < 2:
                continue

            # --- Apply Grubbs' test and optionally exclude outliers ---
            if grubbs_outliers.get("perform_grubbs"):
                outlier_alerts = grubbs_test(group[analyte])
                outlier_indices = outlier_alerts["Outlier Indices"]

                if grubbs_outliers.get("exclude_grubbs") and outlier_indices:
                    group = group.drop(index=outlier_indices)

            if group.empty or len(group) < 2:
                continue  # Skip if not enough data after outlier removal

            # --- Stats Calculation ---
            overall_mean = round(group[analyte].mean(), 2)
            sd = round(group[analyte].std(ddof=1), 2)
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


    # -- Inter-analyser differences --
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
        analyser_comparison,
        filtered_data, 
        outlier_indices 
    )

# --- File Upload ---
with st.expander("📤 Upload Your CSV File", expanded=True):
    st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
    uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])
    
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            st.stop()
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()

    st.subheader("📖 Data Preview")
    st.dataframe(df.head(5), use_container_width=True)
    required_columns = ['Material', 'Analyser', 'Test']
    if not all(col in df.columns for col in required_columns):
        st.error(f"The uploaded file must contain the columns: {', '.join(required_columns)}.")
        st.stop()

    if len(df.columns) <= 5:
        st.warning("❗️ Not enough analyte columns detected.")
    else:
        # Select analyte for analysis
        analyte_options = df.columns[7:]
        selected_analyte = st.selectbox("🔎 Select Analyte to View", analyte_options)

        # --- Westgard & Grubbs Controls Section (Below Analyte Dropdown) ---
        with st.expander("⚙️ Settings: Westgard Rules & Outlier Detection", expanded=True):

            
            tab1, tab2 = st.tabs(["Westgard Rules", "Perform Grubbs` Test"])

            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    rule_1_2s = st.checkbox("1-2s (warning)", value=True)
                    rule_2_2s = st.checkbox("2-2s", value=False)
                    rule_4_1s = st.checkbox("4-1s", value=False)
                    rule_7T = st.checkbox("7T (trend)", value=False)
                with col2:
                    rule_1_3s = st.checkbox("1-3s", value=False)
                    rule_R_4s = st.checkbox("R-4s", value=False)
                    rule_10x = st.checkbox("10x", value=False)
                    rule_8x = st.checkbox("8x", value=False)
            with tab2:
                perform_grubbs = st.checkbox("🔍 Perform Grubbs' test to identify outliers", value=False)
                exclude_grubbs = False
                if perform_grubbs:
                    exclude_grubbs = st.checkbox("🚫 Exclude outliers from calculations", value=False)

            rules_enabled = {
                '1_2s': rule_1_2s,
                '1_3s': rule_1_3s,
                '2_2s': rule_2_2s,
                'R_4s': rule_R_4s,
                '4_1s': rule_4_1s,
                '10x': rule_10x,
                '7T': rule_7T,
                '8x': rule_8x
            }

            grubbs_outliers = {
                "perform_grubbs": perform_grubbs,
                "exclude_grubbs": exclude_grubbs
            }

      
        # Analyze the filtered data
        with st.spinner("Analyzing..."):
            intra_well_df, intra_batch_df, inter_batch_df, diff_df, analyser_comparison, filtered_data, outlier_indices = precision_studies(df, selected_analyte, rules_enabled, grubbs_outliers)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # --- Results Output  ---
        with st.expander("📊 View Results", expanded=True):
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

# --- Optional Reference Section ---
with st.expander("📚 References"):
    st.markdown("""
    **Westgard, J.O., Barry, P.L., and Hunt, M.R. (1981)**, *A Multi-Rule Shewhart Chart for Quality Control in Clinical Chemistry*, Clinical Chemistry, 27 (3), pp.493-501
    (https://westgard.com/downloads/papers-downloads/27-westgard-rules-paper/file.html
    \n **Westgard, J.O., and Barry, P.L.** (1986) Improving Quality Control by use of Multirule Control Procedures. Chapter 4 in Cost-Effective Quality Control: Managing the quality and productivity of analytical processes. AACC Press, Washington, DC, 1986, pp.92-117
    \n **Westgard J.O., and Klee, G.G.** (1994) Quality Management. Chapter 17 in Textbook of Clinical Chemistry, 2nd edition. Burtis C, ed., WB Saunders Company, Philadelphia, pp.548-592.
    \n **Westgard J.O., and Klee, G.G.** (1996) Quality Management. Chapter 16 in Fundamentals of Clinical Chemistry, 4th edition. Burtis C, ed., WB Saunders Company, Philadelphia, 1996, pp.211-223.
    """)

