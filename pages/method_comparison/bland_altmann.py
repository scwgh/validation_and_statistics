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
        \n The limits of agreement help assess whether the differences between two methods are practically significant. If the differences follow an approximately normal distribution, around 95% of the differences should fall within these limits. If the limits of agreement are considered clinically insignificant, the two measurement methods may be regarded as equivalent for practical purposes. However, especially with small sample sizes, these limits may not be reliable. In such cases, the confidence limits for the limits of agreement can provide an indication of the uncertainty. While these confidence limits are only approximate, they should be sufficient for most applications.
                    \n Any results which are identified as outliers will be marked with a purple square (🟪). To exclude outliers from analysis for a given analyte, select the checkbox at the top of the page.""")


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

        # Outlier detection (1.5 x IQR)
        Q1, Q3 = np.percentile(diffs, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        is_outlier = (diffs < lower_bound) | (diffs > upper_bound)

        # Notify user if no outliers are present
        if not is_outlier.any():
            st.info("✅ No outliers detected.")
        else:
            # Display a checkbox to exclude outliers
            st.error("⚠️ Outliers detected.")
            exclude_outliers = st.checkbox("Exclude outliers from analysis", value=False)

            if exclude_outliers:
                # Exclude outliers if checkbox is checked
                valid_indices = diffs[(diffs >= lower_bound) & (diffs <= upper_bound)].index
                vals1, vals2 = vals1.loc[valid_indices], vals2.loc[valid_indices]

                # Show warning and outlier information
                outliers = diffs[is_outlier]
                outlier_ids = df1.loc[outliers.index, 'Sample ID'].tolist()
                if outlier_ids:
                    st.warning(f"⚠️ Outliers excluded: Sample IDs: {', '.join(map(str, outlier_ids))}")

        # else:
        #     valid_indices = diffs.index
        #     st.info("No outliers excluded from analysis.")

        # Re-calculate statistics (mean difference, LoA, p-value)
        diffs = vals1 - vals2
        means = (vals1 + vals2) / 2

        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
        n = len(diffs)

        # Calculate confidence intervals for LoA
        se = std_diff / np.sqrt(n)
        ci_range = 1.96 * se
        ci_upper_upper = loa_upper + ci_range
        ci_upper_lower = loa_upper - ci_range
        ci_lower_upper = loa_lower + ci_range
        ci_lower_lower = loa_lower - ci_range


        # Paired t-test (without outliers if checkbox is checked)
        t_stat, p_val = stats.ttest_rel(vals1, vals2)       

        # --- Plot: Numerical Differences ---
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=means[~is_outlier],  
            y=diffs[~is_outlier],
            mode='markers',
            marker=dict(color='dimgray', symbol='circle'),
            name='Sample',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>Diff: %{y:.3f}<extra></extra>',
            text=df1['Sample ID'][:min_len][~is_outlier] 
        ))
        fig1.add_trace(go.Scatter(
            x=means[is_outlier],  # Outlier points
            y=diffs[is_outlier],
            mode='markers',
            marker=dict(color='deeppink', symbol='square'),
            name='Outlier',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>Diff: %{y:.3f}<extra></extra>',
            text=df1['Sample ID'][:min_len][is_outlier]  
        ))

        # # Shading: Confidence Interval around Upper LoA (light blue)
        # fig1.add_trace(go.Scatter(
        #     x=np.concatenate([means, means[::-1]]),
        #     y=np.concatenate([np.full_like(means, ci_upper_upper), np.full_like(means, ci_upper_lower)[::-1]]),
        #     fill='toself',
        #     fillcolor='rgba(173, 216, 230, 0.3)',  # Light blue
        #     line=dict(color='rgba(255,255,255,0)'),
        #     hoverinfo='skip',
        #     showlegend=True,
        #     name='CI: Upper LoA'
        # ))

        # # Shading: Confidence Interval around Lower LoA (light blue)
        # fig1.add_trace(go.Scatter(
        #     x=np.concatenate([means, means[::-1]]),
        #     y=np.concatenate([np.full_like(means, ci_lower_upper), np.full_like(means, ci_lower_lower)[::-1]]),
        #     fill='toself',
        #     fillcolor='rgba(173, 216, 230, 0.3)',  # Light blue
        #     line=dict(color='rgba(255,255,255,0)'),
        #     hoverinfo='skip',
        #     showlegend=True,
        #     name='CI: Lower LoA'
        # ))

        # # Shading: Overall Limits of Agreement (pale green)
        # fig1.add_trace(go.Scatter(
        #     x=np.concatenate([means, means[::-1]]),
        #     y=np.concatenate([np.full_like(means, loa_upper), np.full_like(means, loa_lower)[::-1]]),
        #     fill='toself',
        #     fillcolor='rgba(144, 238, 144, 0.3)',  # Pale green
        #     line=dict(color='rgba(255,255,255,0)'),
        #     hoverinfo='skip',
        #     showlegend=True,
        #     name='Limits of Agreement'
        # ))

        # Adding reference lines for Mean Diff and LoA
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
            yaxis_title="Difference",
            template="plotly_white"
        )

        # Display plot
        st.plotly_chart(fig1, use_container_width=True)

        # --- Plot: Percentage Differences ---
        # Calculate mean and standard deviation for the percentage differences
        mean_percent_diff = np.mean(percent_diffs)
        std_percent_diff = np.std(percent_diffs, ddof=1)

        # Upper and lower limits of agreement for percentage differences
        loa_upper_percent_diff = mean_percent_diff + 1.96 * std_percent_diff
        loa_lower_percent_diff = mean_percent_diff - 1.96 * std_percent_diff

        # Confidence intervals for % LoA
        se_percent = std_percent_diff / np.sqrt(n)
        ci_range_percent = 1.96 * se_percent
        ci_upper_upper_pct = loa_upper_percent_diff + ci_range_percent
        ci_upper_lower_pct = loa_upper_percent_diff - ci_range_percent
        ci_lower_upper_pct = loa_lower_percent_diff + ci_range_percent
        ci_lower_lower_pct = loa_lower_percent_diff - ci_range_percent

        
        fig2 = go.Figure()

        # Add scatter plot for percentage differences (for non-outliers)
        fig2.add_trace(go.Scatter(
            x=means[~is_outlier],  # Non-outlier points
            y=percent_diffs[~is_outlier],
            mode='markers',
            marker=dict(color='dimgray', symbol='circle'),
            name='Sample',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>% Diff: %{y:.3f}<extra></extra>',
            text=df1['Sample ID'][:min_len][~is_outlier]  # Adding Sample ID as a text attribute
        ))

        # Add scatter plot for outliers in percentage differences
        fig2.add_trace(go.Scatter(
            x=means[is_outlier],  # Outlier points
            y=percent_diffs[is_outlier],
            mode='markers',
            marker=dict(color='deeppink', symbol='square'),
            name='Outlier',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>% Diff: %{y:.3f}<extra></extra>',
            text=df1['Sample ID'][:min_len][is_outlier]  # Adding Sample ID as a text attribute
        ))

        # Mean Percent Difference Line
        fig2.add_trace(go.Scatter(
            x=[means.min(), means.max()],
            y=[mean_percent_diff, mean_percent_diff],
            mode='lines',
            line=dict(color='darkslateblue'),
            name=f"Mean % Diff: {mean_percent_diff:.2f}%"
        ))

        # Upper % LoA
        fig2.add_trace(go.Scatter(
            x=[means.min(), means.max()],
            y=[loa_upper_percent_diff, loa_upper_percent_diff],
            mode='lines',
            line=dict(color='slateblue', dash='dash'),
            name=f"+1.96 SD: {loa_upper_percent_diff:.2f}%"
        ))

        # Lower % LoA
        fig2.add_trace(go.Scatter(
            x=[means.min(), means.max()],
            y=[loa_lower_percent_diff, loa_lower_percent_diff],
            mode='lines',
            line=dict(color='slateblue', dash='dash'),
            name=f"-1.96 SD: {loa_lower_percent_diff:.2f}%"
        ))


        # Update layout for the plot
        fig2.update_layout(
            title=f"{selected_analyte} - Bland-Altmann Plot (% Difference)",
            xaxis_title="Mean of Two Analyzers",
            yaxis_title="Percentage Difference (%)",
            template="plotly_white"
        )

        # Display the plot
        st.plotly_chart(fig2, use_container_width=True)

    # # Display the p-value after re-processing the statistics
    #     st.info(f"P-Value (with outliers{' excluded' if exclude_outliers else ' included'}): {p_val:.5f}")

        # # Display p-value message based on significance
        # if p_val <= 0.05:
        #     st.error(f"🔬 Statistically significant difference between the two analyzers (p ≤ 0.05). P-Value (with outliers{' excluded' if exclude_outliers else ' included'}): {p_val:.5f}")
        # else:
        #     st.success("✅ No statistically significant difference between the two analyzers (p > 0.05).")

        # --- Full Summary Table: All Materials × All Analytes ---
        st.markdown("### 📊 Bland-Altmann Statistical Summary")

        summary_table = []

        for material in df['Material'].unique():
            # Select analytes from columns with index 7 onwards
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

            # Sort by Material, then p-value
            summary_df.sort_values(by=["Material", "p-value"], inplace=True)

            # Highlight statistically different rows
            def highlight_significant(row):
                return ['background-color: #f7f7f2' if row['p-value'] <= 0.00 else ' ' for _ in row]

            st.dataframe(summary_df.style.apply(highlight_significant, axis=1), use_container_width=True)

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
            # Adjusted to select analytes from columns starting at index 7
            analytes = df.columns[7:]
            selected_analyte = st.selectbox("Select Analyte", analytes)

            bland_altmann_analysis(df, material_type, selected_analyte)
