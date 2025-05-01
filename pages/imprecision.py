import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
from datetime import datetime
from utils import apply_app_styling

def check_westgard_rules(values, mean, sd, rules_enabled):
    alerts = []
    n = len(values)
    
    for i in range(n):
        val = values[i]
        z = (val - mean) / sd if sd != 0 else 0

        # 1_2s
        if rules_enabled['1_2s'] and abs(z) > 2:
            alerts.append((i, "1_2s"))

        # 1_3s
        if rules_enabled['1_3s'] and abs(z) > 3:
            alerts.append((i, "1_3s"))

        # 2_2s
        if rules_enabled['2_2s'] and i >= 1:
            prev_z = (values[i-1] - mean) / sd
            if abs(z) > 2 and abs(prev_z) > 2 and np.sign(z) == np.sign(prev_z):
                alerts.append((i-1, "2_2s"))
                alerts.append((i, "2_2s"))

        # R_4s
        if rules_enabled['R_4s'] and i >= 1:
            prev_z = (values[i-1] - mean) / sd
            if (z - prev_z) > 4:
                alerts.append((i-1, "R_4s"))
                alerts.append((i, "R_4s"))

        # 4_1s
        if rules_enabled['4_1s'] and i >= 3:
            zs = [(values[j] - mean) / sd for j in range(i-3, i+1)]
            if all(abs(zj) > 1 and np.sign(zj) == np.sign(zs[0]) for zj in zs):
                alerts.extend([(j, "4_1s") for j in range(i-3, i+1)])

        # 10x
        if rules_enabled['10x'] and i >= 9:
            zs = [(values[j] - mean) / sd for j in range(i-9, i+1)]
            if all(np.sign(zj) == np.sign(zs[0]) for zj in zs):
                alerts.extend([(j, "10x") for j in range(i-9, i+1)])

    return list(set(alerts))  # unique alerts


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
    Imprecision analysis is used to evaluate random error associated with a measurement. Measurement of laboratory analytical error falls into two main categories: \n "***systematic error***" and "***random error***". 
    \n **Systematic errors** are predictable problems influencing observations consistently in one direction, while **random errors** are more unpredictable. Systematic errors are typically demonstrated  by bias (for example, ongoing negative bias for a QC), while random errors by the imprecision measured by the coefficient of variation (CV, %) (for example, one measurement outside of the expected range with all other repeat measurements within range). 
    \n Imprecision affects the reproducibility and repeatability of results. Reproducibility is defined as the closeness of the results of successive measurements under changed conditions. Repeatability is the closeness of the results of at least twenty successive measurements under similar conditions. By contrast, bias is the average deviation from a true value with minimal contribution of imprecision while inaccuracy is the deviation of a single measurement from the true value with significant contribution by imprecision. Multiple measurements, at least twenty and preferably forty, are therefore required for calculating imprecision as well as bias.
    \n ****Why do we perform imprecision analysis?****
    - Verifying analytical precision for method validation.
    - Assessing consistency of quality control materials.
    - Comparing instrument or assay performance.
                
    \n ****What types of imprecision are we interested in?****
    - **Intra-well imprecision**: Variation in repeated measurements within a single well or sample.
    - **Intra-batch imprecision**: Variation in repeated measurements within a single bathch.
    - **Inter-batch imprecision**: Variation in measurements of the same sample across different batches run across days.

    

    >> 💡  Aim for %CVs within your lab's acceptable performance limits (e.g., <5% or <10% depending on the analyte).
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

# --- Instructions ---
with st.expander("📘 Instructions:", expanded=False): 
    st.markdown("""
    This tool allows you to assess **intra-well, intra-batch and inter-batch imprecision** across different levels of control or patient materials.

    To get started:

    1. **Upload your CSV file** – it should contain repeated measurements for the same sample/material across different runs or days.
    2. Make sure your file includes:
    - `Material` (e.g., Control, Patient)
    - `QC Level` (e.g., QC1, QC2, QC3),
    - `Analyser` (e.g., Analyser1, Analyser2)
    - `Run` or `Day` - including in either short or long date format
    - One or more **analyte columns**. Please ensure your analyte names are consistent across the file.
    3. Once uploaded, the app will:
    - Group data by `Material`, `QC Level`, and `Analyte`
    - Calculate intra-batch, inter-batch, and total imprecision. 
        - Intra-well imprecision will also be calculated if provided.
    - Output summaries and visualizations for each analyte
    4. Westgard rules are active on the plots. Use the toggle in the sidebar to activate/deactivate rules. 

    \n > ℹ️ Results are reported in terms of **%CV (Coefficient of Variation)**, which reflects variability relative to the mean.

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

# --- Define Westgard Rules ---
st.sidebar.header("📑 Toggle Westgard Rules")
rules_enabled = {
    '1_2s': st.sidebar.checkbox("1-2s Rule (warning)", value=True),
    '1_3s': st.sidebar.checkbox("1-3s Rule", value=True),
    '2_2s': st.sidebar.checkbox("2-2s Rule", value=True),
    'R_4s': st.sidebar.checkbox("R-4s Rule", value=True),
    '4_1s': st.sidebar.checkbox("4-1s Rule", value=True),
    '10x': st.sidebar.checkbox("10x Rule", value=True),
    '7T': st.sidebar.checkbox("7T Rule (Trend)", value=True),
    '8x': st.sidebar.checkbox("8x Rule", value=True)
}

import numpy as np

def check_westgard_rules(values, mean, sd, rules_enabled):
    alerts = []
    n = len(values)

    if sd == 0:
        return []

    zscores = [(v - mean) / sd for v in values]

    def already_flagged(index, rule):
        return (index, rule) in alerts

    for i in range(n):
        z = zscores[i]

        # 1_2s: 1 point outside ±2 SD (warning)
        if rules_enabled.get('1_2s') and abs(z) > 2 and not already_flagged(i, '1_2s'):
            alerts.append((i, '1_2s'))

        # 1_3s: 1 point outside ±3 SD (rejection)
        if rules_enabled.get('1_3s') and abs(z) > 3 and not already_flagged(i, '1_3s'):
            alerts.append((i, '1_3s'))

        # 2_2s: 2 consecutive points > 2 SD in same direction
        if rules_enabled.get('2_2s') and i >= 1:
            if all(abs(zscores[j]) > 2 for j in [i-1, i]) and np.sign(zscores[i-1]) == np.sign(zscores[i]):
                for j in [i-1, i]:
                    if not already_flagged(j, '2_2s'):
                        alerts.append((j, '2_2s'))

        # R_4s: 2 consecutive results differ by ≥4 SD in opposite directions
        if rules_enabled.get('R_4s') and i >= 1:
            if abs(zscores[i] - zscores[i-1]) >= 4 and np.sign(zscores[i]) != np.sign(zscores[i-1]):
                for j in [i-1, i]:
                    if not already_flagged(j, 'R_4s'):
                        alerts.append((j, 'R_4s'))

        # 4_1s: 4 consecutive points >1 SD in same direction
        if rules_enabled.get('4_1s') and i >= 3:
            segment = zscores[i-3:i+1]
            if all(abs(z) > 1 for z in segment) and len(set(np.sign(z) for z in segment)) == 1:
                for j in range(i-3, i+1):
                    if not already_flagged(j, '4_1s'):
                        alerts.append((j, '4_1s'))

        # 10x: 10 consecutive points on same side of mean
        if rules_enabled.get('10x') and i >= 9:
            segment = zscores[i-9:i+1]
            if all(np.sign(z) == np.sign(segment[0]) and np.sign(z) != 0 for z in segment):
                for j in range(i-9, i+1):
                    if not already_flagged(j, '10x'):
                        alerts.append((j, '10x'))

        # 7T: 7 consecutive values increasing or decreasing
        if rules_enabled.get('7T') and i >= 6:
            diffs = np.diff(values[i-6:i+1])
            if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
                for j in range(i-6, i+1):
                    if not already_flagged(j, '7T'):
                        alerts.append((j, '7T'))

        # 8x: 8 consecutive on one side of mean (redundant with 10x, but stricter)
        if rules_enabled.get('8x') and i >= 7:
            segment = zscores[i-7:i+1]
            if all(np.sign(z) == np.sign(segment[0]) and np.sign(z) != 0 for z in segment):
                for j in range(i-7, i+1):
                    if not already_flagged(j, '8x'):
                        alerts.append((j, '8x'))

    return alerts



# --- Precision Study Function ---
def precision_studies(df, selected_analyte, rules_enabled):
    results = []
    qc_df = df[df['Material'].str.startswith('QC', na=False)]

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

        # --- Plot Traces ---
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

        fig.update_layout(
        hoverlabel=dict(
        font_size=12,  # Increase size
            )
        )


        # --- Westgard Alerts ---
        rule_alerts = check_westgard_rules(group[selected_analyte].tolist(), overall_mean, sd, rules_enabled)
        for i, rule in rule_alerts:
            fig.add_trace(go.Scatter(
                x=[group['Date'].iloc[i]],
                y=[group[selected_analyte].iloc[i]],
                mode='markers',
                marker=dict(color='crimson', size=10, symbol='x'),
                name=f'Violation: {rule}',
                showlegend=False
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

    # for the purposes of HT1 evaluation, we can change the 'Analyser' column to dictate the NBS laboratoratory
        # go back to og notebook and check the code to overlay results 
        # also have a look at the enbs pop

# --- File Upload (Wrapped in Expander) ---
with st.expander("📤 Upload Your CSV File", expanded=True):
    st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
    uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])
    
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.subheader("📖 Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    

    if len(df.columns) <= 5:
        st.warning("❗️ Not enough analyte columns detected.")
    else:
        analyte_options = df.columns[5:]
        selected_analyte = st.selectbox("🔎 Select Analyte to View", analyte_options)

        with st.spinner("Analyzing..."):
            intra_well_df, intra_batch_df, inter_batch_df, diff_df, analyser_comparison = precision_studies(df, selected_analyte, rules_enabled)

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

# --- Optional Reference Section ---
with st.expander("📚 References"):
    st.markdown("""
    **Westgard, J.O., Barry, P.L., and Hunt, M.R. (1981)**, *A Multi-Rule Shewhart Chart for Quality Control in Clinical Chemistry*, Clinical Chemistry, 27 (3), pp.493-501
    (https://westgard.com/downloads/papers-downloads/27-westgard-rules-paper/file.html
    \n **Westgard, J.O., and Barry, P.L.** (1986) Improving Quality Control by use of Multirule Control Procedures. Chapter 4 in Cost-Effective Quality Control: Managing the quality and productivity of analytical processes. AACC Press, Washington, DC, 1986, pp.92-117
    \n **Westgard J.O., and Klee, G.G.** (1994) Quality Management. Chapter 17 in Textbook of Clinical Chemistry, 2nd edition. Burtis C, ed., WB Saunders Company, Philadelphia, pp.548-592.
    \n **Westgard J.O., and Klee, G.G.** (1996) Quality Management. Chapter 16 in Fundamentals of Clinical Chemistry, 4th edition. Burtis C, ed., WB Saunders Company, Philadelphia, 1996, pp.211-223.
    """)
