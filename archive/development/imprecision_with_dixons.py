import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
from datetime import datetime
from utils import apply_app_styling
from scipy import stats

import numpy as np

def dixons_q_test(values, alpha=0.05):
    """
    Performs Dixon's Q test for outlier detection.
    Assumes sorted data and alpha significance level (default is 0.05).
    Returns the index of the outlier if one exists, otherwise returns None.
    """
    n = len(values)
    if n < 3:  # Dixon's Q test needs at least 3 data points
        return None
    
    # Sort the data to apply Dixon's Q test
    values_sorted = np.sort(values)
    
    # Calculate Q statistic for the first and last values
    q_first = (values_sorted[1] - values_sorted[0]) / (values_sorted[-1] - values_sorted[0])
    q_last = (values_sorted[-1] - values_sorted[-2]) / (values_sorted[-1] - values_sorted[0])
    
    # Get critical Q value based on the sample size and alpha level
    critical_q = get_critical_q(n, alpha)
    
    # Compare the Q value with the critical Q value
    if q_first > critical_q:
        return 0  # Outlier is the first element
    if q_last > critical_q:
        return n - 1  # Outlier is the last element
    
    return None

def get_critical_q(n, alpha):
    """
    Returns the critical Q value based on the sample size n and significance level alpha.
    """
    # Predefined critical Q values for various sample sizes (at alpha = 0.05)
    q_values = {
        3: 0.94, 4: 0.76, 5: 0.63, 6: 0.53, 7: 0.45, 8: 0.40,
        9: 0.36, 10: 0.32, 11: 0.30, 12: 0.27, 13: 0.26, 14: 0.24,
        15: 0.23, 16: 0.22, 17: 0.21, 18: 0.20, 19: 0.19, 20: 0.18,
        21: 0.18, 22: 0.17, 23: 0.17, 24: 0.16, 25: 0.16
    }
    # If n is greater than 25, return a fixed critical Q value.
    return q_values.get(n, 0.15)


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
st.title("📊 Imprecision Analysis (with Westgard Rules and Dixon's Q-Test)")

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

    >> 💡  Aim for %CVs within your lab's acceptable performance limits (e.g., <5% or <10% depending on the analyte).
                
    \n From this module, you can visualise and evaluate your data, apply Westgard rules, and perform a Dixon's Q-test to identify statistically significant outliers.  If you would like to perform a **Grubbs' Test** to identify statistically significant outliers from normally distributed data, please navigate to the `Outlier` page.
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

    st.markdown("Grubb's test is used to identify outliers which are statistically significant")

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

with st.expander("ℹ️ What is Dixon's Q-Test?"):
    st.markdown("""
  
    """)

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
    4. Use the toggle in the sidebar to activate/deactivate Westgard rules. 
    5. From the checkbox in the sidebar, perform a Dixon's Q-Test to identify outliers which are statistically significant. 

    \n > ℹ️ Results are reported in terms of **%CV (Coefficient of Variation)**, which reflects variability relative to the mean.

    """)

# --- Define Westgard Rules ---
st.sidebar.markdown(
    "<h3>📑 Apply Westgard Rules</h3>",
    unsafe_allow_html=True
)
rules_enabled = {
    '1_2s': st.sidebar.checkbox("1-2s Rule (warning)", value=True),
    '1_3s': st.sidebar.checkbox("1-3s Rule", value=False),
    '2_2s': st.sidebar.checkbox("2-2s Rule", value=False),
    'R_4s': st.sidebar.checkbox("R-4s Rule", value=False),
    '4_1s': st.sidebar.checkbox("4-1s Rule", value=False),
    '10x': st.sidebar.checkbox("10x Rule", value=False),
    '7T': st.sidebar.checkbox("7T Rule (Trend)", value=False),
    '8x': st.sidebar.checkbox("8x Rule", value=False)
}

# Sidebar for Outliers Identification and Exclusion using Dixon's Q test
st.sidebar.markdown(
    "<h3>❌ Outliers </h3>",
    unsafe_allow_html=True
)
dixons_q_outliers = {
    'dixons_q': st.sidebar.checkbox("Identify outliers using Dixon's Q test", value=False)
}

# Option to exclude outliers from dataset if Dixon's Q test is selected
if dixons_q_outliers['dixons_q']:
    q_threshold = st.sidebar.slider(
        "Q Threshold", min_value=0.1, max_value=0.5, value=0.25, step=0.05,
        help="Select the Dixon's Q threshold for identifying outliers (e.g., 0.25 for the critical Q value)."
    )
    
    exclude_outliers = {
        'exclude_dixons_q': st.sidebar.checkbox("Exclude outliers from dataset", value=False)
    }



def precision_studies(df, selected_analyte, rules_enabled, dixons_q_outliers):
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

    # Step 3: Loop through each group
    for (material, analyzer), group in inter_batch_groups:
        group = group.copy()
        group['Date'] = pd.to_datetime(group['Date'], errors='coerce', dayfirst=True)
        group = group.dropna(subset=['Date', selected_analyte])

        if group.empty or len(group) < 2:
            continue

        # Recalculate overall statistics and plot
        overall_mean = round(group[selected_analyte].mean(), 2)
        sd = round(group[selected_analyte].std(), 2)

        # Apply 7-day moving average
        group = group.sort_values('Date')
        group['Moving Average'] = group[selected_analyte].rolling(window=7, min_periods=1).mean()

        # --- Plot traces ---
        fig.add_trace(go.Scatter(
            x=group['Date'], y=group[selected_analyte], mode='markers',
            marker=dict(color='darkblue', size=6, opacity=0.6),
            name='Sample', showlegend=True
        ), row=row, col=col)

        # fig.add_trace(go.Scatter(
        #     x=group['Date'], y=group['Moving Average'], mode='lines',
        #     line=dict(color='orange', width=2, dash='dot'),
        #     name='7-day MA', showlegend=False
        # ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=group['Date'], y=[overall_mean] * len(group),
            mode='lines', line=dict(color='blue', dash='solid'),
            name='Mean', showlegend=True
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=group['Date'], y=[overall_mean + 2 * sd] * len(group),
            mode='lines', line=dict(color='green', dash='dash'),
            name='+2SD', showlegend=True
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=group['Date'], y=[overall_mean - 2 * sd] * len(group),
            mode='lines', line=dict(color='green', dash='dash'),
            name='-2SD', showlegend=True
        ), row=row, col=col)

        fig.add_trace(go.Scatter(
            x=group['Date'], y=[overall_mean + 3 * sd] * len(group),
            mode='lines', line=dict(color='red', dash='dash'),
            name='+3SD', showlegend=True
        ), row=row, col=col)


        fig.add_trace(go.Scatter(
            x=group['Date'], y=[overall_mean - 3 * sd] * len(group),
            mode='lines', line=dict(color='red', dash='dash'),
            name='-3SD', showlegend=True
        ), row=row, col=col)

        fig.update_layout(
        hoverlabel=dict(
        font_size=14,  # Increase size
            )
        )

        # Dixon's Q test for outliers
        if dixons_q_outliers['dixons_q']:
            outlier_indices = dixons_q_test(group[selected_analyte].values, alpha=0.05)
            
            if outlier_indices:
                # Mark outliers as orange squares
                for idx in outlier_indices:
                    fig.add_trace(go.Scatter(
                        x=[group['Date'].iloc[idx]], y=[group[selected_analyte].iloc[idx]],
                        mode='markers', marker=dict(color='orange', size=10, symbol='square'),
                        name='Dixon\'s Q Outlier', showlegend=True
                    ), row=row, col=col)

                # Display outlier details
                outlier_details = []
                for idx in outlier_indices:
                    analyte_value = group[selected_analyte].iloc[idx]
                    material = group['Material'].iloc[idx]
                    outlier_details.append(f"Analyte: {selected_analyte}, Value: {analyte_value}, Material: {material}")

                outlier_summary = " | ".join(outlier_details)
                st.success(f"Dixon's Q Test applied. {len(outlier_indices)} outliers identified: {outlier_summary}")
        
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
        st.info("No data available for plotting.")        

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
        analyser_comparison,
        filtered_data, 
        outlier_indices 
    )

# --- File Upload (Wrapped in Expander) ---
with st.expander("📤 Upload Your CSV File", expanded=True):
    st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
    uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])
    
if uploaded_file:
    # Read the uploaded file
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
    st.dataframe(df.head(20), use_container_width=True)

    # Validate required columns
    required_columns = ['Material', 'Analyser', 'Test']
    if not all(col in df.columns for col in required_columns):
        st.error(f"The uploaded file must contain the columns: {', '.join(required_columns)}.")
        st.stop()

    if len(df.columns) <= 5:
        st.warning("❗️ Not enough analyte columns detected.")
    else:
        # Select analyte for analysis
        analyte_options = df.columns[5:]
        selected_analyte = st.selectbox("🔎 Select Analyte to View", analyte_options)
      
        # Analyze the filtered data
        with st.spinner("Analyzing..."):
            intra_well_df, intra_batch_df, inter_batch_df, diff_df, analyser_comparison, filtered_data, outlier_indices = precision_studies(df, selected_analyte, rules_enabled, dixons_q_outliers)
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