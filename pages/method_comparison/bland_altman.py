import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from utils import apply_app_styling, units_list
import io

# Set up the page styling
apply_app_styling()

def grubbs_test(values, alpha=0.05):
    import numpy as np
    import pandas as pd
    from scipy import stats

    values = pd.Series(values)  # Convert to Series for idxmax
    n = len(values)
    if n < 3:
        return np.array([False] * n)

    abs_diff = abs(values - values.mean())
    max_diff_idx = abs_diff.idxmax()
    G = abs_diff[max_diff_idx] / values.std(ddof=1)

    # Critical value from Grubbs test table (two-sided)
    t_crit = stats.t.ppf(1 - alpha / (2 * n), df=n - 2)
    G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

    is_outlier = np.array([False] * n)
    if G > G_crit:
        is_outlier[max_diff_idx] = True
    return is_outlier


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
        \n Any results which are identified as outliers using the **Grubbs test** will be marked with a red square (🟥). 
        \n To exclude outliers from analysis for a given analyte, select the checkbox in the Analysis Settings section.""")

    with st.expander("📘 Instructions:"):
        st.markdown("""
        1. **Upload your CSV file** containing multi-analyte results.
        2. Your file must include these columns: `Material`, `Analyser`, `Sample ID`, `Batch ID`, `Lot Number` and at least one analyte.
        3. Configure your analysis settings in the "Analysis Settings" section below.
        4. Click **"Run Bland-Altman Analysis"** to generate plots and statistics for each analyte.
        """)

    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    
        required_cols = ['Analyser', 'Material', 'Sample ID']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns: {', '.join(required_cols)}")
        else:
            # All settings in one expander
            with st.expander("⚙️ Analysis Settings", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    material_type = st.selectbox("Select Material Type", df['Material'].unique())
                    analytes = df.columns[7:]
                    selected_analyte = st.selectbox("Select Analyte", analytes)
                    
                with col2:
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
                
                # Outlier detection settings using Grubbs test
                st.markdown("**Outlier Detection Settings (Grubbs Test)**")
                
                # Add significance level selection for Grubbs test
                alpha = st.selectbox(
                    "Select significance level for Grubbs test",
                    options=[0.05, 0.01, 0.001],
                    index=0,
                    format_func=lambda x: f"α = {x}"
                )
                
                # Pre-calculate outliers for display using Grubbs test
                data = df[df['Material'] == material_type]
                analyte_data = data[['Analyser', 'Sample ID', selected_analyte]].dropna()
                analyte_data[selected_analyte] = pd.to_numeric(analyte_data[selected_analyte], errors='coerce')

                df1 = analyte_data[analyte_data['Analyser'] == analyzer_1].reset_index(drop=True)
                df2 = analyte_data[analyte_data['Analyser'] == analyzer_2].reset_index(drop=True)

                min_len = min(len(df1), len(df2))
                if min_len > 0:
                    vals1 = df1[selected_analyte][:min_len]
                    vals2 = df2[selected_analyte][:min_len]
                    diffs_initial = vals1 - vals2
                    
                    # Apply Grubbs test to differences
                    is_outlier = grubbs_test(diffs_initial.values, alpha=alpha)

                    if not is_outlier.any():
                        st.info("✅ No outliers detected using Grubbs test.")
                        exclude_outliers = False
                    else:
                        # Get outlier details
                        outlier_indices = np.where(is_outlier)[0]
                        outlier_sample_ids = df1.loc[outlier_indices, 'Sample ID'].tolist()
                        outlier_vals1 = vals1.iloc[outlier_indices].tolist()
                        outlier_vals2 = vals2.iloc[outlier_indices].tolist()
                        outlier_diffs = diffs_initial.iloc[outlier_indices].tolist()
                        
                        st.error(f"⚠️ {len(outlier_indices)} outlier(s) detected using Grubbs test (α = {alpha}):")
                        
                        # Create outlier details table
                        outlier_details = []
                        for i, idx in enumerate(outlier_indices):
                            outlier_details.append({
                                'Sample ID': outlier_sample_ids[i],
                                f'{analyzer_1}': round(outlier_vals1[i], 3),
                                f'{analyzer_2}': round(outlier_vals2[i], 3),
                                'Difference': round(outlier_diffs[i], 3),
                                'Mean': round((outlier_vals1[i] + outlier_vals2[i]) / 2, 3)
                            })
                        
                        outlier_df = pd.DataFrame(outlier_details)
                        st.dataframe(outlier_df, use_container_width=True, hide_index=True)
                        
                        exclude_outliers = st.checkbox("Exclude outliers from analysis", value=False)
                        if exclude_outliers:
                            st.warning(f"⚠️ {len(outlier_sample_ids)} outlier(s) will be excluded from analysis: {', '.join(map(str, outlier_sample_ids))}")
                else:
                    exclude_outliers = False
                    is_outlier = np.array([], dtype=bool)

            # Run analysis button
            if st.button("🔬 Run Bland-Altman Analysis", type="primary"):
                bland_altmann_analysis(df, material_type, selected_analyte, analyzer_1, analyzer_2, units, exclude_outliers, is_outlier, alpha)

def bland_altmann_analysis(df, material_type, selected_analyte, analyzer_1, analyzer_2, units, exclude_outliers, is_outlier, alpha):
    data = df[df['Material'] == material_type]

    analyte_data = data[['Analyser', 'Sample ID', selected_analyte]].dropna()
    analyte_data[selected_analyte] = pd.to_numeric(analyte_data[selected_analyte], errors='coerce')

    df1 = analyte_data[analyte_data['Analyser'] == analyzer_1].reset_index(drop=True)
    df2 = analyte_data[analyte_data['Analyser'] == analyzer_2].reset_index(drop=True)

    min_len = min(len(df1), len(df2))
    if min_len == 0:
        st.warning("No overlapping samples between analyzers.")
        return

    vals1 = df1[selected_analyte][:min_len].reset_index(drop=True)
    vals2 = df2[selected_analyte][:min_len].reset_index(drop=True)
    
    # Recalculate outliers using Grubbs test for consistency
    diffs_initial = vals1 - vals2
    is_outlier = grubbs_test(diffs_initial.values, alpha=alpha)
    
    # Apply outlier filtering if requested
    if exclude_outliers and is_outlier.any():
        valid_indices = ~is_outlier
        vals1 = vals1[valid_indices].reset_index(drop=True)
        vals2 = vals2[valid_indices].reset_index(drop=True)
        is_outlier = np.array([False] * len(vals1))  # Reset outlier mask after filtering
    
    means = (vals1 + vals2) / 2
    N = len(means)
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
    
    # Create masks for plotting
    if exclude_outliers:
        # If outliers are excluded, all remaining points are normal
        normal_indices = list(range(len(means)))
        outlier_indices = []
    else:
        # If outliers are not excluded, separate normal and outlier points
        normal_indices = [i for i in range(len(means)) if i < len(is_outlier) and not is_outlier[i]]
        outlier_indices = [i for i in range(len(means)) if i < len(is_outlier) and is_outlier[i]]
    
    # Plot normal points
    if normal_indices:
        normal_means = means.iloc[normal_indices]
        normal_diffs = diffs.iloc[normal_indices]
        normal_sample_ids = df1['Sample ID'].iloc[normal_indices] if not exclude_outliers else df1['Sample ID'].iloc[:len(means)]
        
        fig1.add_trace(go.Scatter(
            x=normal_means,  
            y=normal_diffs,
            mode='markers',
            marker=dict(color='mediumblue', symbol='circle', size=8),
            name=f'N = {N}',  
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>Diff: %{y:.3f}<extra></extra>',
            text=normal_sample_ids
        ))
    
    # Plot outlier points (only if not excluded)
    if outlier_indices and not exclude_outliers:
        outlier_means = means.iloc[outlier_indices]
        outlier_diffs = diffs.iloc[outlier_indices]
        outlier_sample_ids = df1['Sample ID'].iloc[outlier_indices]
        
        fig1.add_trace(go.Scatter(
            x=outlier_means,  
            y=outlier_diffs,
            mode='markers',
            marker=dict(color='red', symbol='square', size=8),
            name='Outlier (Grubbs)',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>Diff: %{y:.3f}<extra></extra>',
            text=outlier_sample_ids
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
    
    # Plot normal points
    if normal_indices:
        normal_percent_diffs = percent_diffs.iloc[normal_indices]
        
        fig2.add_trace(go.Scatter(
            x=normal_means,
            y=normal_percent_diffs,
            mode='markers',
            marker=dict(color='mediumblue', symbol='circle', size=8),
            name=f'N = {N}', 
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>% Diff: %{y:.3f}<extra></extra>',
            text=normal_sample_ids
        ))

    # Plot outlier points (only if not excluded)
    if outlier_indices and not exclude_outliers:
        outlier_percent_diffs = percent_diffs.iloc[outlier_indices]
        
        fig2.add_trace(go.Scatter(
            x=outlier_means,
            y=outlier_percent_diffs,
            mode='markers',
            marker=dict(color='red', symbol='square', size=8),
            name='Outlier (Grubbs)',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>% Diff: %{y:.3f}<extra></extra>',
            text=outlier_sample_ids
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
        name=f'N = {N}', 
        text=df1['Sample ID'].iloc[:len(vals1)],
        hovertemplate='<b>Sample ID: %{text}</b><br>%{x:.2f} vs %{y:.2f}<extra></extra>'
    ))

# Plot outlier points (only if not excluded)
    if outlier_indices and not exclude_outliers:
        outlier_percent_diffs = percent_diffs.iloc[outlier_indices]
        
        fig_reg.add_trace(go.Scatter(
            x=outlier_means,
            y=outlier_percent_diffs,
            mode='markers',
            marker=dict(color='red', symbol='square', size=8),
            name='Outlier (Grubbs)',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>% Diff: %{y:.3f}<extra></extra>',
            text=outlier_sample_ids
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
        name=f'N = {N}'  
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

        for analyte in analytes:
            try:
                data = df[df['Material'] == material][['Analyser', 'Sample ID', analyte]].dropna()
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

                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs, ddof=1)
                loa_upper = mean_diff + 1.96 * std_diff
                loa_lower = mean_diff - 1.96 * std_diff
                _, p_val = stats.ttest_rel(vals1, vals2)

                summary_table.append({
                'Material': material,
                'Analyte': analyte,
                'N Samples': min_len,
                'Mean Difference': round(mean_diff, 3),
                'SD of Differences': round(std_diff, 3),
                'LoA Lower': round(loa_lower, 3),
                'LoA Upper': round(loa_upper, 3),
                'p-value': round(p_val, 3)
            })
            except Exception as e:
                st.warning(f"⚠️ Could not process '{analyte}' for material '{material}': {e}")

    if summary_table:
        summary_df = pd.DataFrame(summary_table)
        st.dataframe(summary_df, use_container_width=True)

    with st.expander("📚 References"):
        st.markdown("""
        **Giavarina, D. (2015)**, *Limit of Blank, Limit of Detection and Limit of Quantitation*, Understanding Bland Altman analysis. Biochemia medica, 25(2), pp. 141–151. https://doi.org/10.11613/BM.2015.015 """)