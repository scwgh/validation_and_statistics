import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from utils import apply_app_styling
import io

# Set up the page styling
apply_app_styling()

# Run function for method_comparison.py to call
def run():
    st.title("🥼 Bland-Altmann Analysis")

    with st.expander("📘 What is Bland-Altman Analysis?"):
        st.markdown("""
        Bland-Altman analysis is a method for assessing the **agreement between two measurement methods** by plotting the **difference** against the **mean** of the two methods for each sample.
        \n Given a set of paired measurements (_x_ and _y_) where _i_ = 1 to _n_, the Bland-Altmann plots calculate as:
        """)
        st.latex(r'\text{y axis} = {y}_i - {x}_i')
        st.latex(r'\text{x axis} = \frac{{y}_i + {x}_i}{2}')
        st.markdown("""
        This analysis is used to evaluate if two analyzers provide results that are **consistently close** enough for clinical or research purposes.
        \n The reference line for the mean gives an indication of the bias between the two methods. 
                    \n The limits of agreement help assess whether the differences between two methods are practically significant. If the differences follow an approximately normal distribution, around 95% of the differences should fall within these limits. If the limits of agreement are considered clinically insignificant, the two measurement methods may be regarded as equivalent for practical purposes. However, especially with small sample sizes, these limits may not be reliable. In such cases, the confidence limits for the limits of agreement can provide an indication of the uncertainty. While these confidence limits are only approximate, they should be sufficient for most applications.""")

    with st.expander("📘 Instructions:"):
        st.markdown("""
        1. **Upload your CSV file** containing multi-analyte results.
        2. Your file must include these columns: `Material`, `Analyser`, `Sample ID`, and at least one analyte.
        3. Select **two analyzers** to compare from the dropdowns.
        4. Click **"Run Bland-Altman Analysis"** to generate plots and statistics for each analyte.
        """)

    # Function to perform Bland-Altmann analysis
    def bland_altmann_analysis(df, material_type, selected_analyte):
        data = df[df['Material'] == material_type]

        analyzers = data['Analyser'].unique()
        if len(analyzers) < 2:
            st.warning("Need at least two analyzers for comparison.")
            return

        analyzer_1, analyzer_2 = analyzers[:2]

        analyte_data = data[['Analyser', 'Sample ID', selected_analyte]].dropna()
        analyte_data[selected_analyte] = pd.to_numeric(analyte_data[selected_analyte], errors='coerce')

        df1 = analyte_data[analyte_data['Analyser'] == analyzer_1].reset_index(drop=True)
        df2 = analyte_data[analyte_data['Analyser'] == analyzer_2].reset_index(drop=True)

        min_len = min(len(df1), len(df2))
        if min_len == 0:
            st.warning("No overlapping samples between analyzers.")
            return

        vals1 = df1[selected_analyte][:min_len]
        vals2 = df2[selected_analyte][:min_len]

        # Bland-Altmann calculations
        means = (vals1 + vals2) / 2
        diffs = vals1 - vals2
        percent_diffs = (diffs / ((vals1 + vals2) / 2).replace(0, np.nan)) * 100

        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
        n = len(diffs)

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(vals1, vals2)

        # --- Plot: Numerical Differences ---
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=means, 
            y=diffs, 
            mode='markers',
            marker=dict(color='dimgray', symbol='square'),
            name='Sample',
            hovertemplate='Mean: %{x:.2f}<br>Diff: %{y:.2f}<extra></extra>'
        ))

        fig1.add_hline(y=mean_diff, line_color='blue', annotation_text=f"Mean Diff: {mean_diff:.2f}", annotation_position="top left")
        fig1.add_hline(y=loa_upper, line_color='blue', line_dash='dash', annotation_text=f"+1.96 SD: {loa_upper:.2f}")
        fig1.add_hline(y=loa_lower, line_color='blue', line_dash='dash', annotation_text=f"-1.96 SD: {loa_lower:.2f}")

        fig1.update_layout(
            title=f"{selected_analyte} - Bland-Altmann Plot (Numerical Difference)",
            xaxis_title="Mean of Two Analyzers",
            yaxis_title="Difference",
            template="plotly_white"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # --- Plot: Percentage Differences ---
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=means / 2, 
            y=percent_diffs, 
            mode='markers',
            marker=dict(color='dimgray', symbol='square'),
            name='Sample',
            hovertemplate='Mean/2: %{x:.2f}<br>% Diff: %{y:.2f}<extra></extra>'
        ))
        fig2.add_hline(y=0, line_color='grey')
        fig2.add_hline(y=np.mean(percent_diffs), line_color='blue',
                       annotation_text=f"Mean % Diff: {np.mean(percent_diffs):.2f}", annotation_position="top left")
        fig2.add_hline(y=np.mean(percent_diffs) + 1.96 * np.std(percent_diffs, ddof=1), line_color='blue', line_dash='dash')
        fig2.add_hline(y=np.mean(percent_diffs) - 1.96 * np.std(percent_diffs, ddof=1), line_color='blue', line_dash='dash')

        fig2.update_layout(
            title=f"{selected_analyte} - Bland-Altmann Plot (% Difference)",
            xaxis_title="Mean of Two Analyzers / 2",
            yaxis_title="Percentage Difference (%)",
            template="plotly_white"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # --- Summary ---
        st.markdown("### 📊 Statistical Summary")
        st.markdown(f"""
        - **Analyzers Compared**: `{analyzer_1}` vs `{analyzer_2}`  
        - **Number of Samples**: {n}  
        - **Mean Difference**: {mean_diff:.2f}  
        - **Standard Deviation of Differences**: {std_diff:.2f}  
        - **95% Limits of Agreement (LoA)**: [{loa_lower:.2f}, {loa_upper:.2f}]  
        - **Paired t-test p-value**: {p_val:.4f}  
        """)

        if p_val > 0.05:
            st.error("🔬 Statistically significant difference between the two analyzers (p > 0.05).")
        else:
            st.success("✅ No statistically significant difference between the two analyzers (p ≤ 0.05).")

    # --- File Upload ---
    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        required_cols = ['Analyser', 'Material', 'Sample ID']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns: {', '.join(required_cols)}")
        else:
            material_type = st.selectbox("Select Material Type", df['Material'].unique())
            analytes = [col for col in df.columns if col not in required_cols + ['Date', 'Test']]
            selected_analyte = st.selectbox("Select Analyte", analytes)

            bland_altmann_analysis(df, material_type, selected_analyte)


# --- Run Bland-Altman for ALL analytes and download ---
    if uploaded_file and 'df' in locals():
        st.markdown("### 📁 Download Bland-Altman Results for All Analytes")

    if st.button("📊 Run for All Analytes"):
        analytes = [col for col in df.columns if col not in ['Analyser', 'Material', 'Sample ID', 'Date', 'Test']]
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            for analyte in analytes:
                try:
                    data = df[df['Material'] == material_type][['Analyser', 'Sample ID', analyte]].dropna()
                    data[analyte] = pd.to_numeric(data[analyte], errors='coerce')

                    analyzers = data['Analyser'].unique()
                    if len(analyzers) < 2:
                        continue

                    df1 = data[data['Analyser'] == analyzers[0]].reset_index(drop=True)
                    df2 = data[data['Analyser'] == analyzers[1]].reset_index(drop=True)
                    min_len = min(len(df1), len(df2))
                    if min_len == 0:
                        continue

                    vals1 = df1[analyte][:min_len]
                    vals2 = df2[analyte][:min_len]
                    diffs = vals1 - vals2
                    means = (vals1 + vals2) / 2
                    percent_diffs = (diffs / means.replace(0, np.nan)) * 100

                    mean_diff = np.mean(diffs)
                    std_diff = np.std(diffs, ddof=1)
                    loa_upper = mean_diff + 1.96 * std_diff
                    loa_lower = mean_diff - 1.96 * std_diff
                    _, p_val = stats.ttest_rel(vals1, vals2)

                    summary_df = pd.DataFrame({
                        'Sample ID': df1['Sample ID'][:min_len],
                        'Analyzer 1': vals1,
                        'Analyzer 2': vals2,
                        'Mean': means,
                        'Difference': diffs,
                        '% Difference': percent_diffs
                    })

                    stats_row = pd.DataFrame([{
                        'Sample ID': 'Summary',
                        'Analyzer 1': '',
                        'Analyzer 2': '',
                        'Mean': f"{mean_diff:.2f}",
                        'Difference': f"LoA: {loa_lower:.2f} to {loa_upper:.2f}",
                        '% Difference': f"p={p_val:.4f}"
                    }])

                    final_df = pd.concat([summary_df, stats_row])
                    final_df.to_excel(writer, sheet_name=analyte[:31], index=False)
                except Exception as e:
                    st.warning(f"Failed for {analyte}: {e}")

        output.seek(0)
        st.download_button(
            label="⬇️ Download Excel File",
            data=output,
            file_name="bland_altman_all_analytes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
