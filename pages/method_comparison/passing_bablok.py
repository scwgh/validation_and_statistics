import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import linregress
from utils import apply_app_styling, units_list

# === Utility Functions ===
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

def calculate_r2(x, y, slope, intercept):
    """Calculate R-squared value for regression"""
    y_pred = slope * x + intercept
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def passing_bablok_regression(x, y):
    """Perform Passing-Bablok regression"""
    n = len(x)
    if n >500:
        st.warning("⚠️ Too many data points for Passing-Bablok regression. Consider reducing the dataset size.")
        return 1.0, 0.0
    slopes = []

    # Compute all pairwise slopes
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            if dx != 0:
                slopes.append(dy / dx)

    if len(slopes) == 0:
        return 1.0, 0.0  # Default values if no slopes can be calculated

    slopes = np.sort(slopes)
    n_slopes = len(slopes)

    # Get median slope
    if n_slopes % 2 == 1:
        slope = slopes[n_slopes // 2]
    else:
        slope = 0.5 * (slopes[n_slopes // 2 - 1] + slopes[n_slopes // 2])

    # Calculate intercept as median of y - slope * x
    intercepts = y - slope * x
    intercept = np.median(intercepts)

    return slope, intercept

def detect_outliers_grubbs(data, alpha=0.05):
    """Detect outliers using Grubbs test - returns indices of outliers"""
    try:
        outlier_mask = grubbs_test(data, alpha=alpha)
        return np.where(outlier_mask)[0].tolist()  # Return indices instead of boolean array
    except Exception as e:
        st.warning(f"Grubbs test failed: {str(e)}")
        return []

def check_outliers_for_analyte(df, analyte, analyzer_1, analyzer_2, alpha=0.05):
    """Check for outliers in a specific analyte without removing them"""
    if analyte not in df.columns:
        return None

    filtered = df[df['Analyser'].isin([analyzer_1, analyzer_2])]
    values_1 = filtered[filtered['Analyser'] == analyzer_1][['Sample ID', analyte]].dropna()
    values_2 = filtered[filtered['Analyser'] == analyzer_2][['Sample ID', analyte]].dropna()

    merged = pd.merge(values_1, values_2, on='Sample ID', suffixes=('_1', '_2'))
    if merged.shape[0] < 2:
        return None

    # Convert to numeric and handle errors
    x = pd.to_numeric(merged[f'{analyte}_1'], errors='coerce')
    y = pd.to_numeric(merged[f'{analyte}_2'], errors='coerce')
    sample_ids = merged['Sample ID']

    # Remove rows with NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    sample_ids = sample_ids[valid_mask].reset_index(drop=True)

    if len(x) < 2:
        return None
    
    # Detect outliers in both x and y data
    outliers_x = detect_outliers_grubbs(x.values, alpha)
    outliers_y = detect_outliers_grubbs(y.values, alpha)
    
    # Combine outlier indices
    outlier_indices = set(outliers_x + outliers_y)
    outlier_indices = {idx for idx in outlier_indices if 0 <= idx < len(x)}
    
    if outlier_indices:
        outlier_samples = sample_ids.iloc[list(outlier_indices)].tolist()
        return {
            'total_outliers': len(outlier_indices),
            'outlier_samples': outlier_samples,
            'outlier_indices': list(outlier_indices)
        }
    
    return None

def perform_analysis(df, analyte, analyzer_1, analyzer_2, units, remove_outliers=False, alpha=0.05):
    """Perform Passing-Bablok analysis for a single analyte"""
    if analyte not in df.columns:
        st.warning(f"⚠ {analyte} column not found in data.")
        return None, None, None, None

    filtered = df[df['Analyser'].isin([analyzer_1, analyzer_2])]

    values_1 = filtered[filtered['Analyser'] == analyzer_1][['Sample ID', analyte]].dropna()
    values_2 = filtered[filtered['Analyser'] == analyzer_2][['Sample ID', analyte]].dropna()

    merged = pd.merge(values_1, values_2, on='Sample ID', suffixes=('_1', '_2'))
    if merged.shape[0] < 2:
        return None, None, None, None

    # Convert to numeric and handle errors
    x = pd.to_numeric(merged[f'{analyte}_1'], errors='coerce')
    y = pd.to_numeric(merged[f'{analyte}_2'], errors='coerce')
    sample_ids = merged['Sample ID']

    # Remove rows with NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    sample_ids = sample_ids[valid_mask].reset_index(drop=True)

    if len(x) < 2:
        return None, None, None, None
    
    # Always detect outliers for plotting purposes
    outliers_x = detect_outliers_grubbs(x.values, alpha)
    outliers_y = detect_outliers_grubbs(y.values, alpha)
    
    # Combine outlier indices
    outlier_indices = set(outliers_x + outliers_y)
    outlier_indices = {idx for idx in outlier_indices if 0 <= idx < len(x)}
    
    outlier_info = None
    x_analysis = x.copy()
    y_analysis = y.copy()
    sample_ids_analysis = sample_ids.copy()
    
    if outlier_indices:
        outlier_info = {
            'total_outliers': len(outlier_indices),
            'outlier_samples': sample_ids.iloc[list(outlier_indices)].tolist(),
            'outlier_indices': list(outlier_indices)
        }
        
        if remove_outliers:
            # Remove outliers for analysis
            keep_indices = [i for i in range(len(x)) if i not in outlier_indices]
            x_analysis = x.iloc[keep_indices].reset_index(drop=True)
            y_analysis = y.iloc[keep_indices].reset_index(drop=True)
            sample_ids_analysis = sample_ids.iloc[keep_indices].reset_index(drop=True)
            
            if len(x_analysis) < 2:
                st.warning(f"⚠ Too few data points remaining after outlier removal for {analyte}")
                return None, None, None, outlier_info

    slope, intercept = passing_bablok_regression(x_analysis.values, y_analysis.values)
    r2 = calculate_r2(x_analysis.values, y_analysis.values, slope, intercept)

    results = {
        "Analyte": analyte,
        "Analyzer 1": analyzer_1,
        "Analyzer 2": analyzer_2,
        "Slope": round(slope, 4),
        "Intercept": round(intercept, 4),
        "R²": round(r2, 4),
        "n": len(x_analysis)
    }

    # Pass all data for plotting (including outliers)
    fig = plot_regression_plotly(
        analyte, x, y, sample_ids, slope, intercept, r2, 
        analyzer_1, analyzer_2, units, list(outlier_indices) if outlier_indices else []
    )
    
    # Create merged dataframe for display (all data)
    merged_display = pd.DataFrame({
        'Sample ID': sample_ids,
        f'{analyte}_1': x,
        f'{analyte}_2': y
    })
    
    return results, fig, merged_display, outlier_info

def plot_regression_plotly(analyte, x_data, y_data, sample_ids, slope, intercept, r2, analyzer_1, analyzer_2, units, outlier_indices=[], remove_outliers=False):
    """Create Plotly regression plot with outliers highlighted or removed"""
    if len(x_data) == 0:
        return go.Figure()
    
    # Convert to numpy arrays for easier indexing
    x_array = np.array(x_data)
    y_array = np.array(y_data)
    sample_ids_array = np.array(sample_ids)
    
    # Create masks for outliers and normal points
    outlier_mask = np.isin(range(len(x_array)), outlier_indices)
    normal_mask = ~outlier_mask
    
    # If removing outliers, only use normal points for plotting
    if remove_outliers and len(outlier_indices) > 0:
        x_plot = x_array[normal_mask]
        y_plot = y_array[normal_mask]
        sample_ids_plot = sample_ids_array[normal_mask]
        
        x_line = np.linspace(min(x_plot), max(x_plot), 100) if len(x_plot) > 0 else np.array([])
        y_line = slope * x_line + intercept
        
        # Calculate confidence interval using only normal points
        residuals = y_plot - (slope * x_plot + intercept) if len(x_plot) > 0 else np.array([])
        y_err = 1.96 * np.std(residuals) if len(residuals) > 1 else 0
        
        fig = go.Figure()
        
        # Add only normal data points
        if len(x_plot) > 0:
            fig.add_trace(go.Scatter(
                x=x_plot,
                y=y_plot,
                mode='markers',
                marker=dict(color="mediumslateblue", size=8, symbol='circle'),
                text=sample_ids_plot,
                hovertemplate='<b>Sample ID:</b> %{text}<br><b>X:</b> %{x:.2f}<br><b>Y:</b> %{y:.2f}<extra></extra>',
                name="Data Points"
            ))
    else:
        # Show all points including outliers
        x_line = np.linspace(min(x_array), max(x_array), 100)
        y_line = slope * x_line + intercept
        
        # Calculate confidence interval
        residuals = y_array - (slope * x_array + intercept)
        y_err = 1.96 * np.std(residuals) if len(residuals) > 1 else 0

        fig = go.Figure()

        # Add normal data points first
        if np.any(normal_mask):
            fig.add_trace(go.Scatter(
                x=x_array[normal_mask],
                y=y_array[normal_mask],
                mode='markers',
                marker=dict(color="mediumslateblue", size=8, symbol='circle'),
                text=sample_ids_array[normal_mask],
                hovertemplate='<b>Sample ID:</b> %{text}<br><b>X:</b> %{x:.2f}<br><b>Y:</b> %{y:.2f}<extra></extra>',
                name="Data Points"
            ))

        # Add outlier points with red squares
        if len(outlier_indices) > 0 and np.any(outlier_mask):
            fig.add_trace(go.Scatter(
                x=x_array[outlier_mask],
                y=y_array[outlier_mask],
                mode='markers',
                marker=dict(color="red", size=10, symbol='square'),
                text=sample_ids_array[outlier_mask],
                hovertemplate='<b>Sample ID:</b> %{text}<br><b>X:</b> %{x:.2f}<br><b>Y:</b> %{y:.2f}<br><b>Status:</b> Outlier<extra></extra>',
                name="Outliers"
            ))

    # Add regression line
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        line=dict(color='crimson', width=2),
        name="Regression Line"
    ))

    # Add confidence interval
    if y_err > 0:
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_line, x_line[::-1]]),
            y=np.concatenate([y_line - y_err, (y_line + y_err)[::-1]]),
            fill='toself',
            fillcolor='rgba(220, 20, 60, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name="95% CI"
        ))

    # Add equation and R² to legend as invisible trace
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)', size=0),
        showlegend=True,
        name=f"y = {slope:.4f}x + {intercept:.4f}<br>R² = {r2:.4f}, n = {len(x_array)}",
        hoverinfo='skip'
    ))

    fig.update_layout(
        title=f"Passing-Bablok Regression for {analyte}",
        xaxis_title=f"{analyzer_1} ({units})",
        yaxis_title=f"{analyzer_2} ({units})",
        plot_bgcolor='white',
        showlegend=True,
        title_font=dict(size=16),
    )

    return fig

# === Streamlit App ===

apply_app_styling()

st.title("📊 Passing-Bablok Comparison")

def passing_bablok():
    with st.expander("📘 What is Passing Bablok Regression?"):
        st.markdown("""
        **Passing Bablok regression** is a **non-parametric method comparison technique** that is robust to outliers and does not assume a specific error distribution.
        
        - Uses median-based calculations instead of least squares
        - Suitable for method comparison studies
        - Does not assume normal distribution of errors
        - Provides slope and intercept estimates with confidence intervals
        """)

    with st.expander("📘 Instructions:"):
        st.markdown("""
        1. Upload a CSV file containing `Date`, `Test`, `Analyser`, `Material`, `Sample ID`, `Batch ID`, `Lot Number`, and analyte columns.
        2. Configure analysis settings in the Settings section below.
        3. Select the two analyzers you want to compare.
        4. Optionally enable outlier detection using Grubss` test and set parameters. Any outliers detected will be highlighted on the plots with a red square (🟥).
        5. If outliers exclusion is enabled, they will be removed from the regression analysis.
        6. View regression plots and statistics.
        """)

    with st.expander("📤 Upload CSV File", expanded=True):
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="uploader")

    # Initialize session state variables
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'analyzers' not in st.session_state:
        st.session_state.analyzers = []

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            required_cols = ['Analyser', 'Material', 'Sample ID']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return
            
            st.success(f"✅ File uploaded successfully!")
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return

    if st.session_state.df is not None:
        df = st.session_state.df
        
        # === SETTINGS SECTION IN EXPANDER ===
        with st.expander("⚙️ Analysis Settings", expanded=True):
            # Material selection
            st.subheader("📋 Material Selection")
            materials = sorted(df['Material'].dropna().unique().tolist())
            material_options = ['All'] + materials
            selected_material = st.selectbox("Select Material", material_options)

            # Filter by material
            if selected_material != "All":
                df_filtered = df[df['Material'] == selected_material].copy()
            else:
                df_filtered = df.copy()

            # Analyzer selection
            st.subheader("🔬 Analyzer Selection")
            analyzers = sorted(df_filtered['Analyser'].dropna().unique())
            st.session_state.analyzers = analyzers
            
            if len(analyzers) < 2:
                st.error("At least two analyzers are required for comparison.")
                return

            col1, col2 = st.columns(2)
            with col1:
                analyzer_1 = st.selectbox("Select Reference Analyzer", analyzers)
            with col2:
                analyzer_2 = st.selectbox("Select Test Analyzer", [a for a in analyzers if a != analyzer_1])

            # Units selection
            st.subheader("📏 Units Configuration")
            units = st.selectbox(
                "Select Units for Analytes",
                options=units_list, 
                index=0
            )

            # Get analyte columns for outlier checking
            metadata_cols = ['Date', 'Test', 'Analyser', 'Material', 'Sample ID', 'Batch ID', 'Lot Number']
            analyte_cols = [col for col in df_filtered.columns if col not in metadata_cols]
            
            # Outlier Detection Settings
            st.subheader("🎯 Outlier Detection Settings")
            
            # Outlier controls - stacked vertically
            alpha = st.slider("Significance level (α)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
            remove_outliers = st.checkbox("Exclude outliers from regression analysis", value=False)
            
            # Check for outliers in all analytes
            outlier_warnings = []
            
            for analyte in analyte_cols:
                outlier_check = check_outliers_for_analyte(df_filtered, analyte, analyzer_1, analyzer_2, alpha)
                if outlier_check and outlier_check['total_outliers'] > 0:
                    outlier_warnings.append({
                        'analyte': analyte,
                        'count': outlier_check['total_outliers'],
                        'samples': outlier_check['outlier_samples']
                    })
            
            # Display outlier status
            if outlier_warnings:
                st.warning(f"⚠️ **{len(outlier_warnings)} analyte(s) have outliers detected** (α = {alpha})")
                
                # Use checkbox to show/hide details
                show_outlier_details = st.checkbox("Show outlier details", value=False)
                
                if show_outlier_details:
                    st.markdown("**Outlier Details:**")
                    for warning in outlier_warnings:
                        st.write(f"• **{warning['analyte']}**: {warning['count']} outliers - {', '.join(warning['samples'])}")
            else:
                st.info(f"ℹ️ **No outliers detected** for any analyte in {selected_material} material (α = {alpha})")    
                
        # Run analysis button
        if st.button("🚀 Run Passing Bablok Regression", type="primary"):
            with st.spinner("Running analysis..."):
                st.write("### 📋 Data Preview")
                st.dataframe(df_filtered.head())
                
                results = []
                outlier_summary = []
                
                for analyte in analyte_cols:
                    st.write(f"### 📊 Analysis for {analyte}")
                    
                    result, fig, merged, outlier_info = perform_analysis(
                        df_filtered, analyte, analyzer_1, analyzer_2, units, remove_outliers, alpha
                    )
                    
                    if result:
                        results.append(result)
                        
                        # Display outlier information if applicable
                        if outlier_info and outlier_info['total_outliers'] > 0:
                            if remove_outliers:
                                st.info(f"ℹ️ **{outlier_info['total_outliers']} outlier(s) excluded from regression analysis for {analyte}:** {', '.join(outlier_info['outlier_samples'])}")
                            else:
                                st.info(f"ℹ️ **{outlier_info['total_outliers']} outlier(s) detected for {analyte}:** {', '.join(outlier_info['outlier_samples'])} (included in analysis)")
                            
                            outlier_summary.append({
                                'Analyte': analyte,
                                'Outliers_Detected': outlier_info['total_outliers'],
                                'Outlier_Samples': ', '.join(outlier_info['outlier_samples']),
                                'Excluded_from_Analysis': 'Yes' if remove_outliers else 'No'
                            })
                        
                        # Display plot
                        st.plotly_chart(fig, use_container_width=True)

                        # Display data table
                        with st.expander(f"📊 Data for {analyte}"):
                            # Calculate percentage difference
                            merged_display = merged.copy()
                            merged_display["% Difference"] = (
                                (merged_display[f"{analyte}_2"] - merged_display[f"{analyte}_1"]) / merged_display[f"{analyte}_1"]
                            ) * 100
                            merged_display["% Difference"] = merged_display["% Difference"].round(2)

                            # Add outlier status column if there are outliers
                            if outlier_info and outlier_info['total_outliers'] > 0:
                                merged_display["Outlier"] = merged_display['Sample ID'].isin(outlier_info['outlier_samples'])
                                merged_display["Outlier"] = merged_display["Outlier"].map({True: "Yes", False: "No"})

                            # Rename columns for display
                            display_columns = ["Sample ID", analyzer_1, analyzer_2, "% Difference"]
                            if "Outlier" in merged_display.columns:
                                display_columns.append("Outlier")
                            
                            display_df = merged_display.rename(columns={
                                f"{analyte}_1": analyzer_1,
                                f"{analyte}_2": analyzer_2
                            })[display_columns]
                            
                            st.dataframe(display_df)
                    else:
                        st.warning(f"⚠ Insufficient data for {analyte}. Skipping...")

                # Display results summary
                if results:
                    st.success("✅ Analysis complete!")
                    st.write("### 📈 Summary Results")
                    result_df = pd.DataFrame(results)
                    st.dataframe(result_df)

                    # Download buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇ Download Results", 
                            csv, 
                            "passing_bablok_results.csv", 
                            "text/csv"
                        )
                    
                    # Download outlier summary if applicable
                    if outlier_summary:
                        with col2:
                            outlier_df = pd.DataFrame(outlier_summary)
                            outlier_csv = outlier_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "⬇ Download Outlier Summary", 
                                outlier_csv, 
                                "outlier_summary.csv", 
                                "text/csv"
                            )
                else:
                    st.warning("No analytes had sufficient data for analysis.")

def run():
    passing_bablok()

if __name__ == "__main__":
    run()