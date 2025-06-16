import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from utils import apply_app_styling, units_list
import io

# Set up the page styling
apply_app_styling()

# Run function for method_comparison.py to call
def run():
    st.title("🥼 Bland-Altman Analysis")

    with st.expander("📘 What is Bland-Altman Analysis?"):
        st.markdown("""
        Bland-Altman analysis is a method for assessing the **agreement between two measurement methods** by plotting the **difference** against the **mean** of the two methods for each sample.
        \n Given a set of paired measurements (_x_ and _y_) where _i_ = 1 to _n_, the Bland-Altman plots calculate as:
        """)
        st.latex(r'\text{y axis} = {y}_i - {x}_i')
        st.latex(r'\text{x axis} = \frac{{y}_i + {x}_i}{2}')
        st.markdown("""
        This analysis is used to evaluate if two analyzers provide results that are **consistently close** enough for clinical or research purposes.
        \n The reference line for the mean gives an indication of the bias between the two methods. 
        \n The limits of agreement help assess whether the differences between two methods are practically significant. If the differences follow an approximately normal distribution, around 95% of the differences should fall within these limits. If the limits of agreement are considered clinically insignificant, the two measurement methods may be regarded as equivalent for practical purposes. However, especially with small sample sizes, these limits may not be reliable. In such cases, the confidence limits for the limits of agreement can provide an indication of the uncertainty. While these confidence limits are only approximate, they should be sufficient for most applications.
        \n Any results which are identified as outliers will be marked with a red square (🟥). 
        \n To exclude outliers from analysis for a given analyte, select the checkbox at the top of the page.""")


    with st.expander("📘 Instructions:"):
        st.markdown("""
        1. **Upload your CSV file** containing multi-analyte results.
        2. Your file must include these columns: `Material`, `Analyser`, `Sample ID`, `Batch ID`, `Lot Number` and at least one analyte.
        3. Select **two analyzers** to compare from the dropdowns.
        4. Click **"Run Bland-Altman Analysis"** to generate plots and statistics for each analyte.
        """)

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
        means = (vals1 + vals2) / 2
                # Detect outliers based on initial diffs
        diffs_initial = vals1 - vals2
        Q1, Q3 = np.percentile(diffs_initial, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        is_outlier = (diffs_initial < lower_bound) | (diffs_initial > upper_bound)

        with st.expander(":gear: Outlier Settings", expanded=True):
            st.markdown(""" """)
            if not is_outlier.any():
                st.info("✅ No outliers detected.")
                valid_indices = diffs_initial.index
            else:
                st.error("⚠️ Outliers detected.")
                exclude_outliers = st.checkbox("Exclude outliers from analysis", value=False)
                if exclude_outliers:
                    valid_indices = diffs_initial[~is_outlier].index
                    outlier_ids = df1.loc[is_outlier, 'Sample ID'].tolist()
                    if outlier_ids:
                        st.warning(f"⚠️ Outliers excluded: Sample IDs: {', '.join(map(str, outlier_ids))}")
                else:
                    valid_indices = diffs_initial.index

        # Filter data
        vals1 = vals1.loc[valid_indices]
        vals2 = vals2.loc[valid_indices]

        # Now recalculate these AFTER outlier filtering
        means = (vals1 + vals2) / 2
        diffs = vals1 - vals2
        percent_diffs = (diffs / ((vals1 + vals2) / 2).replace(0, np.nan)) * 100

        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
        n = len(diffs)
        se = std_diff / np.sqrt(n)
        ci_range = 1.96 * se
        ci_upper_upper = loa_upper + ci_range
        ci_upper_lower = loa_upper - ci_range
        ci_lower_upper = loa_lower + ci_range
        ci_lower_lower = loa_lower - ci_range
        t_stat, p_val = stats.ttest_rel(vals1, vals2)       

        # --- Plot: Numerical Differences ---
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=means[~is_outlier],  
            y=diffs[~is_outlier],
            mode='markers',
            marker=dict(color='dimgray', symbol='circle', size=8),
            name='Sample',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>Diff: %{y:.3f}<extra></extra>',
            text=df1['Sample ID'][:min_len][~is_outlier] 
        ))
        fig1.add_trace(go.Scatter(
            x=means[is_outlier],  # Outlier points
            y=diffs[is_outlier],
            mode='markers',
            marker=dict(color='red', symbol='square', size=8),
            name='Outlier',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>Diff: %{y:.3f}<extra></extra>',
            text=df1['Sample ID'][:min_len][is_outlier]  
        ))
        fig1.add_trace(go.Scatter(
            x=[means.min(), means.max()],
            y=[mean_diff, mean_diff],
            mode='lines',
            line=dict(color='darkslateblue', dash='solid'),
            name=f"Mean Diff: {mean_diff:.2f}"
        ))
        fig1.add_trace(go.Scatter(
            x=[means.min(), means.max()],
            y=[loa_upper, loa_upper],
            mode='lines',
            line=dict(color='slateblue', dash='dash'),
            name=f"+1.96 SD = {loa_upper:.2f}"
        ))
        fig1.add_trace(go.Scatter(
            x=[means.min(), means.max()],
            y=[loa_lower, loa_lower],
            mode='lines',
            line=dict(color='slateblue', dash='dash'),
            name=f"-1.96 SD = {loa_lower:.2f}"
        ))
        fig1.update_layout(
            title=f"{selected_analyte} - Bland-Altmann Plot (Numerical Difference)",
            xaxis_title="Mean of Two Analyzers",
            yaxis_title="Difference (n)",
            template="plotly_white"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # --- Plot: Percentage Differences ---
        mean_percent_diff = np.mean(percent_diffs)
        std_percent_diff = np.std(percent_diffs, ddof=1)
        loa_upper_percent_diff = mean_percent_diff + 1.96 * std_percent_diff
        loa_lower_percent_diff = mean_percent_diff - 1.96 * std_percent_diff
        se_percent = std_percent_diff / np.sqrt(n)
        ci_range_percent = 1.96 * se_percent
        ci_upper_upper_pct = loa_upper_percent_diff + ci_range_percent
        ci_upper_lower_pct = loa_upper_percent_diff - ci_range_percent
        ci_lower_upper_pct = loa_lower_percent_diff + ci_range_percent
        ci_lower_lower_pct = loa_lower_percent_diff - ci_range_percent

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=means[~is_outlier],
            y=percent_diffs[~is_outlier],
            mode='markers',
            marker=dict(color='dimgray', symbol='circle', size=8),
            name='Sample',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>% Diff: %{y:.3f}<extra></extra>',
            text=df1['Sample ID'][:min_len][~is_outlier]
        ))

        fig2.add_trace(go.Scatter(
            x=means[is_outlier],
            y=percent_diffs[is_outlier],
            mode='markers',
            marker=dict(color='deeppink', symbol='square', size=8),
            name='Outlier',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>% Diff: %{y:.3f}<extra></extra>',
            text=df1['Sample ID'][:min_len][is_outlier]
        ))
        fig2.add_trace(go.Scatter(
            x=[means.min(), means.max()],
            y=[mean_percent_diff, mean_percent_diff],
            mode='lines',
            line=dict(color='darkslateblue'),
            name=f"Mean % Diff: {mean_percent_diff:.2f}%"
        ))
        fig2.add_trace(go.Scatter(
            x=[means.min(), means.max()],
            y=[loa_upper_percent_diff, loa_upper_percent_diff],
            mode='lines',
            line=dict(color='slateblue', dash='dash'),
            name=f"+1.96 SD: {loa_upper_percent_diff:.2f}%"
        ))
        fig2.add_trace(go.Scatter(
            x=[means.min(), means.max()],
            y=[loa_lower_percent_diff, loa_lower_percent_diff],
            mode='lines',
            line=dict(color='slateblue', dash='dash'),
            name=f"-1.96 SD: {loa_lower_percent_diff:.2f}%"
        ))
        fig2.update_layout(
            title=f"{selected_analyte} - Bland-Altmann Plot (% Difference)",
            xaxis_title="Mean of Two Analyzers",
            yaxis_title="Percentage Difference (%)",
            template="plotly_white"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # --- Plot 3: Regression Plot ---
        slope, intercept, r_value, p_val_reg, _ = stats.linregress(vals1, vals2)
        x_range = np.linspace(min(vals1.min(), vals2.min()), max(vals1.max(), vals2.max()), 100)
        y_fit = intercept + slope * x_range

        fig_reg = go.Figure()
        fig_reg.add_trace(go.Scatter(
            x=vals1,
            y=vals2,
            mode='markers',
            marker=dict(color='mediumblue', symbol='circle', size=8),
            name='Sample',
            text=df1['Sample ID'].iloc[:len(vals1)],
            hovertemplate='<b>Sample ID: %{text}</b><br>%{x:.2f} vs %{y:.2f}<extra></extra>'
        ))
        fig_reg.add_trace(go.Scatter(
            x=x_range,
            y=y_fit,
            mode='lines',
            line=dict(color='crimson', dash='solid'),
            name=f'Regression Line<br>y = {slope:.2f}x + {intercept:.2f}<br>R² = {r_value**2:.3f}'
        ))
        fig_reg.add_trace(go.Scatter(
            x=x_range,
            y=x_range,
            mode='lines',
            line=dict(color='gray', dash='dot'),
            name='Line of Identity (y = x)'
        ))

        fig_reg.update_layout(
            title=f"{selected_analyte} - Regression Plot",
            xaxis_title=f"{analyzer_1} ({units})",
            yaxis_title=f"{analyzer_2} ({units})",
            template="plotly_white"
        )
        st.plotly_chart(fig_reg, use_container_width=True)

        # --- Plot 4: Histogram of Differences ---
        diffs_df = pd.DataFrame({
            'Sample ID': df1['Sample ID'].iloc[:len(diffs)],
            'Difference': diffs
        })

        diffs_df_sorted = diffs_df.sort_values('Difference').reset_index(drop=True)

        fig_hist_bar = go.Figure()
        fig_hist_bar.add_trace(go.Bar(
            x=diffs_df_sorted.index,
            y=diffs_df_sorted['Difference'],
            marker_color='indianred',
            text=diffs_df_sorted['Sample ID'],
            hovertemplate='<b>Sample ID: %{text}</b><br>Difference: %{y:.3f}<extra></extra>',
            name='Difference'
        ))

        fig_hist_bar.add_hline(y=mean_diff, line=dict(color='black', dash='solid'), annotation_text='Mean Diff', annotation_position='top left')
        fig_hist_bar.add_hline(y=loa_upper, line=dict(color='slateblue', dash='dash'), annotation_text='+1.96 SD', annotation_position='top right')
        fig_hist_bar.add_hline(y=loa_lower, line=dict(color='slateblue', dash='dash'), annotation_text='-1.96 SD', annotation_position='bottom right')

        fig_hist_bar.update_layout(
            title=f"{selected_analyte} - Bar Plot of Differences",
            xaxis_title="Sample Index (Sorted)",
            yaxis_title="Difference (Analyser 1 - Analyser 2)",
            template="plotly_white"
        )
        st.plotly_chart(fig_hist_bar, use_container_width=True)

        # --- Full Summary Table: All Materials × All Analytes ---
        st.markdown("### 📊 Bland-Altmann Statistical Summary")

        summary_table = []

        for material in df['Material'].unique():
            analytes = df.columns[7:]

            for selected_analyte in analytes:
                try:
                    data = df[df['Material'] == material][['Analyser', 'Sample ID', selected_analyte]].dropna()
                    data[selected_analyte] = pd.to_numeric(data[selected_analyte], errors='coerce')

                    analyzers = data['Analyser'].unique()
                    if len(analyzers) < 2:
                        continue

                    df1 = data[data['Analyser'] == analyzers[0]].reset_index(drop=True)
                    df2 = data[data['Analyser'] == analyzers[1]].reset_index(drop=True)
                    min_len = min(len(df1), len(df2))
                    if min_len == 0:
                        continue

                    vals1 = df1[selected_analyte][:min_len]
                    vals2 = df2[selected_analyte][:min_len]
                    diffs = vals1 - vals2
                    means = (vals1 + vals2) / 2

                    mean_diff = np.mean(diffs)
                    std_diff = np.std(diffs, ddof=1)
                    loa_upper = mean_diff + 1.96 * std_diff
                    loa_lower = mean_diff - 1.96 * std_diff
                    _, p_val = stats.ttest_rel(vals1, vals2)

                    outcome = "Statistically significant" if p_val <= 0.05 else "Not statistically significant"

                    summary_table.append({
                    'Material': material,
                    'Analyte': selected_analyte,
                    'N Samples': min_len,
                    'Mean Difference': round(mean_diff, 3),
                    'SD of Differences': round(std_diff, 3),
                    'LoA Lower': round(loa_lower, 3),
                    'LoA Upper': round(loa_upper, 3),
                    'p-value': round(p_val, 3)
                    # 'Outcome': outcome
                })
                except Exception as e:
                    st.warning(f"⚠️ Could not process '{selected_analyte}' for material '{material}': {e}")

        if summary_table:
            summary_df = pd.DataFrame(summary_table)
            st.dataframe(summary_df, use_container_width=True)

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
            analytes = df.columns[7:]
            selected_analyte = st.selectbox("Select Analyte", analytes)
            analyzers = df["Analyser"].unique()
            if len(analyzers) < 2:
                st.warning("Need at least two analyzers in the dataset.")
                return

            analyzer_1 = st.selectbox("Select Reference Analyzer (Analyser 1)", analyzers, key="ref")
            remaining_analyzers = [a for a in analyzers if a != analyzer_1]
            analyzer_2 = st.selectbox("Select Test Analyzer (Analyser 2)", remaining_analyzers, key="test")

            units = st.selectbox(
                "Select Units for Analytes",
                options=units_list, 
                index=0
            )

            bland_altmann_analysis(df, material_type, selected_analyte)