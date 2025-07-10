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
    """
    Perform Grubbs test for outlier detection
    """
    values = pd.Series(values)
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

def prepare_matched_data(df, material_type, selected_analyte, analyzer_1, analyzer_2):
    """
    Prepare matched data for Bland-Altman analysis
    """
    # Filter data for the selected material
    data = df[df['Material'] == material_type].copy()
    
    # Get data for each analyzer
    data_analyzer1 = data[data['Analyser'] == analyzer_1][['Sample ID', selected_analyte]].dropna()
    data_analyzer2 = data[data['Analyser'] == analyzer_2][['Sample ID', selected_analyte]].dropna()
    
    # Convert to numeric
    data_analyzer1[selected_analyte] = pd.to_numeric(data_analyzer1[selected_analyte], errors='coerce')
    data_analyzer2[selected_analyte] = pd.to_numeric(data_analyzer2[selected_analyte], errors='coerce')
    
    # Remove NaN values
    data_analyzer1 = data_analyzer1.dropna()
    data_analyzer2 = data_analyzer2.dropna()
    
    # Merge on Sample ID to get only matching samples
    merged_data = pd.merge(
        data_analyzer1, 
        data_analyzer2, 
        on='Sample ID', 
        suffixes=('_1', '_2'),
        how='inner'
    )
    
    return merged_data

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
                
                # Prepare matched data for outlier detection preview
                merged_data = prepare_matched_data(df, material_type, selected_analyte, analyzer_1, analyzer_2)
                
                if len(merged_data) == 0:
                    st.warning(f"No matching samples found between {analyzer_1} and {analyzer_2} for {selected_analyte}")
                    return
                
                # Extract values and calculate differences
                vals1 = merged_data[f'{selected_analyte}_1']
                vals2 = merged_data[f'{selected_analyte}_2']
                diffs_initial = vals1 - vals2
                
                # Apply Grubbs test to differences
                is_outlier = grubbs_test(diffs_initial.values, alpha=alpha)
                
                exclude_outliers = False
                if not is_outlier.any():
                    st.info("✅ No outliers detected using Grubbs test.")
                else:
                    # Get outlier details
                    outlier_indices = np.where(is_outlier)[0]
                    outlier_sample_ids = merged_data['Sample ID'].iloc[outlier_indices].tolist()
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

            # Run analysis button
            if st.button("🔬 Run Bland-Altman Analysis", type="primary"):
                bland_altman_analysis(df, material_type, selected_analyte, analyzer_1, analyzer_2, units, exclude_outliers, alpha)

def bland_altman_analysis(df, material_type, selected_analyte, analyzer_1, analyzer_2, units, exclude_outliers, alpha):
    """
    Perform Bland-Altman analysis and create plots
    """
    # Prepare matched data
    merged_data = prepare_matched_data(df, material_type, selected_analyte, analyzer_1, analyzer_2)
    
    if len(merged_data) == 0:
        st.warning(f"No matching samples found between {analyzer_1} and {analyzer_2} for {selected_analyte}")
        return
    
    # Extract values and calculate differences
    vals1 = merged_data[f'{selected_analyte}_1']
    vals2 = merged_data[f'{selected_analyte}_2']
    sample_ids = merged_data['Sample ID']
    
    # Calculate differences and means
    diffs = vals1 - vals2
    means = (vals1 + vals2) / 2
    percent_diffs = (diffs / means.replace(0, np.nan)) * 100
    
    # Apply Grubbs test to differences
    is_outlier = grubbs_test(diffs.values, alpha=alpha)
    
    # Apply outlier exclusion logic
    if exclude_outliers and is_outlier.any():
        # Create mask for filtering out outliers
        normal_mask = ~is_outlier
        
        # Filter out outliers from all data (both for analysis and plotting)
        vals1_final = vals1[normal_mask]
        vals2_final = vals2[normal_mask]
        diffs_final = diffs[normal_mask]
        means_final = means[normal_mask]
        percent_diffs_final = percent_diffs[normal_mask]
        sample_ids_final = sample_ids[normal_mask]
        
        analysis_note = f"Analysis performed on {len(vals1_final)} samples (excluded {sum(is_outlier)} outliers)"
        title_suffix = " (Outliers Excluded)"
        
        # For reference, store original data for information purposes
        excluded_samples = sample_ids[is_outlier].tolist()
        
    else:
        # Use all data (outliers included in both analysis and plotting)
        vals1_final = vals1
        vals2_final = vals2
        diffs_final = diffs
        means_final = means
        percent_diffs_final = percent_diffs
        sample_ids_final = sample_ids
        
        analysis_note = f"Analysis performed on {len(vals1_final)} samples"
        title_suffix = ""
        
        # If outliers detected but not excluded, we'll show them highlighted
        if is_outlier.any():
            title_suffix = " (Outliers Highlighted)"
    
    st.info(analysis_note)
    
    # Calculate statistics using final data
    N = len(vals1_final)
    mean_diff = np.mean(diffs_final)
    std_diff = np.std(diffs_final, ddof=1)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    
    # Confidence intervals for limits of agreement
    se = std_diff / np.sqrt(N)
    ci_range = 1.96 * se
    ci_upper_upper = loa_upper + ci_range
    ci_upper_lower = loa_upper - ci_range
    ci_lower_upper = loa_lower + ci_range
    ci_lower_lower = loa_lower - ci_range
    
    # Statistical tests using final data
    t_stat, p_val = stats.ttest_rel(vals1_final, vals2_final)
    slope, intercept, r_value, p_val_reg, _ = stats.linregress(vals1_final, vals2_final)
    
    # Calculate percentage statistics using final data
    mean_percent_diff = np.mean(percent_diffs_final)
    std_percent_diff = np.std(percent_diffs_final, ddof=1)
    loa_upper_percent = mean_percent_diff + 1.96 * std_percent_diff
    loa_lower_percent = mean_percent_diff - 1.96 * std_percent_diff
    
    # Determine plot ranges based on final data (so axes update when outliers excluded)
    x_range = [means_final.min(), means_final.max()]
    x_range_reg = np.linspace(min(vals1_final.min(), vals2_final.min()), 
                             max(vals1_final.max(), vals2_final.max()), 100)
    y_fit = intercept + slope * x_range_reg
    
    # --- Plot 1: Numerical Differences ---
    fig1 = go.Figure()
    
    # Determine how to show points based on outlier settings
    if not exclude_outliers and is_outlier.any():
        # Show outliers highlighted (not excluded)
        normal_mask = ~is_outlier
        
        # Normal points
        if normal_mask.any():
            fig1.add_trace(go.Scatter(
                x=means[normal_mask],
                y=diffs[normal_mask],
                mode='markers',
                marker=dict(color='mediumblue', symbol='circle', size=8),
                name=f'Normal (N = {sum(normal_mask)})',
                hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>Diff: %{y:.3f}<extra></extra>',
                text=sample_ids[normal_mask]
            ))
        
        # Outliers (highlighted but included)
        fig1.add_trace(go.Scatter(
            x=means[is_outlier],
            y=diffs[is_outlier],
            mode='markers',
            marker=dict(color='red', symbol='square', size=8),
            name=f'Outlier (N = {sum(is_outlier)})',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>Diff: %{y:.3f}<extra></extra>',
            text=sample_ids[is_outlier]
        ))
    else:
        # Show only final data (either no outliers or outliers excluded)
        fig1.add_trace(go.Scatter(
            x=means_final,
            y=diffs_final,
            mode='markers',
            marker=dict(color='mediumblue', symbol='circle', size=8),
            name=f'N = {N}',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>Diff: %{y:.3f}<extra></extra>',
            text=sample_ids_final
        ))
    
    # Add reference lines (based on final data)
    fig1.add_trace(go.Scatter(
        x=x_range,
        y=[mean_diff, mean_diff],
        mode='lines',
        line=dict(color='darkslateblue', dash='solid'),
        name=f"Mean Diff: {mean_diff:.3f}"
    ))
    fig1.add_trace(go.Scatter(
        x=x_range,
        y=[loa_upper, loa_upper],
        mode='lines',
        line=dict(color='slateblue', dash='dash'),
        name=f"+1.96 SD: {loa_upper:.3f}"
    ))
    fig1.add_trace(go.Scatter(
        x=x_range,
        y=[loa_lower, loa_lower],
        mode='lines',
        line=dict(color='slateblue', dash='dash'),
        name=f"-1.96 SD: {loa_lower:.3f}"
    ))
    
    fig1.update_layout(
        title=f"{selected_analyte} - Bland-Altman Plot (Numerical Difference){title_suffix}",
        xaxis_title=f"Mean of {analyzer_1} and {analyzer_2} ({units})",
        yaxis_title=f"Difference ({analyzer_1} - {analyzer_2}) ({units})",
        template="plotly_white"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # --- Plot 2: Percentage Differences ---
    fig2 = go.Figure()
    
    # Same logic for percentage plot
    if not exclude_outliers and is_outlier.any():
        # Show outliers highlighted (not excluded)
        normal_mask = ~is_outlier
        
        # Normal points
        if normal_mask.any():
            fig2.add_trace(go.Scatter(
                x=means[normal_mask],
                y=percent_diffs[normal_mask],
                mode='markers',
                marker=dict(color='mediumblue', symbol='circle', size=8),
                name=f'Normal (N = {sum(normal_mask)})',
                hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>% Diff: %{y:.1f}%<extra></extra>',
                text=sample_ids[normal_mask]
            ))
        
        # Outliers (highlighted but included)
        fig2.add_trace(go.Scatter(
            x=means[is_outlier],
            y=percent_diffs[is_outlier],
            mode='markers',
            marker=dict(color='red', symbol='square', size=8),
            name=f'Outlier (N = {sum(is_outlier)})',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>% Diff: %{y:.1f}%<extra></extra>',
            text=sample_ids[is_outlier]
        ))
    else:
        # Show only final data
        fig2.add_trace(go.Scatter(
            x=means_final,
            y=percent_diffs_final,
            mode='markers',
            marker=dict(color='mediumblue', symbol='circle', size=8),
            name=f'N = {N}',
            hovertemplate='<b>Sample ID: %{text}</b><br>Mean: %{x:.3f}<br>% Diff: %{y:.1f}%<extra></extra>',
            text=sample_ids_final
        ))
    
    # Add reference lines (based on final data)
    fig2.add_trace(go.Scatter(
        x=x_range,
        y=[mean_percent_diff, mean_percent_diff],
        mode='lines',
        line=dict(color='darkslateblue', dash='solid'),
        name=f"Mean % Diff: {mean_percent_diff:.2f}%"
    ))
    fig2.add_trace(go.Scatter(
        x=x_range,
        y=[loa_upper_percent, loa_upper_percent],
        mode='lines',
        line=dict(color='slateblue', dash='dash'),
        name=f"+1.96 SD: {loa_upper_percent:.2f}%"
    ))
    fig2.add_trace(go.Scatter(
        x=x_range,
        y=[loa_lower_percent, loa_lower_percent],
        mode='lines',
        line=dict(color='slateblue', dash='dash'),
        name=f"-1.96 SD: {loa_lower_percent:.2f}%"
    ))
    
    fig2.update_layout(
        title=f"{selected_analyte} - Bland-Altman Plot (Percentage Difference){title_suffix}",
        xaxis_title=f"Mean of {analyzer_1} and {analyzer_2} ({units})",
        yaxis_title=f"Percentage Difference (%)",
        template="plotly_white"
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # --- Plot 3: Regression Plot ---
    fig3 = go.Figure()
    
    # Same logic for regression plot
    if not exclude_outliers and is_outlier.any():
        # Show outliers highlighted (not excluded)
        normal_mask = ~is_outlier
        
        # Normal points
        if normal_mask.any():
            fig3.add_trace(go.Scatter(
                x=vals1[normal_mask],
                y=vals2[normal_mask],
                mode='markers',
                marker=dict(color='mediumblue', symbol='circle', size=8),
                name=f'Normal (N = {sum(normal_mask)})',
                text=sample_ids[normal_mask],
                hovertemplate='<b>Sample ID: %{text}</b><br>%{x:.3f} vs %{y:.3f}<extra></extra>'
            ))
        
        # Outliers (highlighted but included)
        fig3.add_trace(go.Scatter(
            x=vals1[is_outlier],
            y=vals2[is_outlier],
            mode='markers',
            marker=dict(color='red', symbol='square', size=8),
            name=f'Outlier (N = {sum(is_outlier)})',
            text=sample_ids[is_outlier],
            hovertemplate='<b>Sample ID: %{text}</b><br>%{x:.3f} vs %{y:.3f}<extra></extra>'
        ))
    else:
        # Show only final data
        fig3.add_trace(go.Scatter(
            x=vals1_final,
            y=vals2_final,
            mode='markers',
            marker=dict(color='mediumblue', symbol='circle', size=8),
            name=f'N = {N}',
            text=sample_ids_final,
            hovertemplate='<b>Sample ID: %{text}</b><br>%{x:.3f} vs %{y:.3f}<extra></extra>'
        ))
    
    # Add regression line (based on final data)
    fig3.add_trace(go.Scatter(
        x=x_range_reg,
        y=y_fit,
        mode='lines',
        line=dict(color='crimson', dash='solid'),
        name=f'Regression Line<br>y = {slope:.3f}x + {intercept:.3f}<br>R² = {r_value**2:.3f}'
    ))
    
    # Add line of identity
    fig3.add_trace(go.Scatter(
        x=x_range_reg,
        y=x_range_reg,
        mode='lines',
        line=dict(color='gray', dash='dot'),
        name='Line of Identity (y = x)'
    ))
    
    fig3.update_layout(
        title=f"{selected_analyte} - Regression Plot{title_suffix}",
        xaxis_title=f"{analyzer_1} ({units})",
        yaxis_title=f"{analyzer_2} ({units})",
        template="plotly_white"
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    # --- Plot 4: Bar Plot of Differences ---
    diffs_df = pd.DataFrame({
        'Sample ID': sample_ids_final,
        'Difference': diffs_final
    })
    diffs_df_sorted = diffs_df.sort_values('Difference').reset_index(drop=True)
    
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=diffs_df_sorted.index,
        y=diffs_df_sorted['Difference'],
        marker_color='indianred',
        text=diffs_df_sorted['Sample ID'],
        hovertemplate='<b>Sample ID: %{text}</b><br>Difference: %{y:.3f}<extra></extra>',
        name=f'N = {N}'
    ))
    
    # Add reference lines (based on final data)
    fig4.add_hline(y=mean_diff, line=dict(color='black', dash='solid'), 
                   annotation_text=f'Mean Diff: {mean_diff:.3f}', annotation_position='top left')
    fig4.add_hline(y=loa_upper, line=dict(color='slateblue', dash='dash'), 
                   annotation_text=f'+1.96 SD: {loa_upper:.3f}', annotation_position='top right')
    fig4.add_hline(y=loa_lower, line=dict(color='slateblue', dash='dash'), 
                   annotation_text=f'-1.96 SD: {loa_lower:.3f}', annotation_position='bottom right')
    
    fig4.update_layout(
        title=f"{selected_analyte} - Bar Plot of Differences{title_suffix}",
        xaxis_title="Sample Index (Sorted by Difference)",
        yaxis_title=f"Difference ({analyzer_1} - {analyzer_2}) ({units})",
        template="plotly_white"
    )
    st.plotly_chart(fig4, use_container_width=True)
    
    # --- Summary Statistics ---
    st.markdown("### 📊 Statistical Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Samples", N)
        st.metric("Mean Difference", f"{mean_diff:.3f} {units}")
        st.metric("SD of Differences", f"{std_diff:.3f} {units}")
    
    with col2:
        st.metric("Lower LoA", f"{loa_lower:.3f} {units}")
        st.metric("Upper LoA", f"{loa_upper:.3f} {units}")
        st.metric("p-value (paired t-test)", f"{p_val:.3f}")
    
    with col3:
        st.metric("Correlation (R²)", f"{r_value**2:.3f}")
        st.metric("Slope", f"{slope:.3f}")
        st.metric("Intercept", f"{intercept:.3f}")
    
    # Add outlier information if present
    if is_outlier.any():
        st.markdown("### 🔍 Outlier Information")
        outlier_status = "Excluded from analysis and plots" if exclude_outliers else "Highlighted in plots but included in analysis"
        st.info(f"**{sum(is_outlier)} outlier(s) detected** using Grubbs test (α = {alpha}). Status: {outlier_status}")
        
        if exclude_outliers:
            st.warning(f"Excluded samples: {', '.join(map(str, excluded_samples))}")
    
    # --- Full Summary Table: All Materials × All Analytes ---
    st.markdown("### 📋 Complete Analysis Summary")
    
    summary_table = []
    
    for material in df['Material'].unique():
        analytes = df.columns[7:]
        
        for analyte in analytes:
            try:
                # Get analyzers for this material
                material_data = df[df['Material'] == material]
                analyzers = material_data['Analyser'].unique()
                
                if len(analyzers) < 2:
                    continue
                
                # Use first two analyzers
                analyzer1, analyzer2 = analyzers[:2]
                
                # Prepare matched data
                matched_data = prepare_matched_data(df, material, analyte, analyzer1, analyzer2)
                
                if len(matched_data) == 0:
                    continue
                
                # Calculate statistics
                vals1 = matched_data[f'{analyte}_1']
                vals2 = matched_data[f'{analyte}_2']
                diffs = vals1 - vals2
                
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs, ddof=1)
                loa_upper = mean_diff + 1.96 * std_diff
                loa_lower = mean_diff - 1.96 * std_diff
                _, p_val = stats.ttest_rel(vals1, vals2)
                
                summary_table.append({
                    'Material': material,
                    'Analyte': analyte,
                    'Analyzer 1': analyzer1,
                    'Analyzer 2': analyzer2,
                    'N Samples': len(matched_data),
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
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Add download button for summary
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Summary as CSV",
            data=csv,
            file_name=f"bland_altman_summary_{material_type}_{selected_analyte}.csv",
            mime="text/csv"
        )
    
    with st.expander("📚 References"):
        st.markdown("""
        **Giavarina, D. (2015)**, *Understanding Bland Altman analysis*. Biochemia medica, 25(2), pp. 141–151. 
        https://doi.org/10.11613/BM.2015.015
        
        **Bland, J. M., & Altman, D. G. (1986)**. *Statistical methods for assessing agreement between two methods of clinical measurement*. The lancet, 1(8476), 307-310.
        """)

if __name__ == "__main__":
    run()