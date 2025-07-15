import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.odr import ODR, RealData, Model
import plotly.graph_objects as go
from utils import apply_app_styling, units_list

def run():
    apply_app_styling()

    st.title("⚖️ Deming Regression Analysis")

    with st.expander("📘 What is Deming Regression?"):
        st.markdown("""
        **Deming regression** is used when **both X and Y variables have measurement error**, often in method comparison studies.
        - **Slope**: Proportional bias (1 = ideal)
        - **Intercept**: Constant bias (0 = ideal)
        - **R²**: Strength of the linear relationship
        """)

    with st.expander("📘 Instructions:"):
        st.markdown("""
        1. Upload a CSV with Material, Analyser, Sample ID, and analyte columns.
        2. Choose two analyzers for comparison.
        3. Choose a material type and a unit.
        4. Run Deming Regression to view plots and download results.
        """)

    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: Material, Analyser, and Sample ID.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"])
        df = pd.read_csv(uploaded_file) if uploaded_file else None

    if df is not None:
        # Step 1: Analyzer selection
        analyzers = df['Analyser'].dropna().unique()
        if len(analyzers) < 2:
            st.error("❌ Need at least two analyzers.")
        else:
            analyzer_1 = st.selectbox("Select Analyzer 1", analyzers, index=0)
            analyzer_2 = st.selectbox("Select Analyzer 2", analyzers, index=1)

            if analyzer_1 == analyzer_2:
                st.warning("⚠ Please select two different analyzers.")
            else:
                # Step 2: Material selection
                valid_materials = ["EQA", "Patient"]
                material_options = df['Material'].dropna().unique()
                filtered_materials = [m for m in material_options if m in valid_materials]
                selected_material = st.selectbox(
                    "Select Material Type",
                    options=filtered_materials,
                    index=0 if "EQA" in filtered_materials else 0
                )

                # Step 3: Analyte selection
                analytes = [col for col in df.columns if col not in ['Date', 'Test', 'Material', 'Analyser', 'Date', 'Sample ID', 'Batch ID', 'Lot Number']]
                selected_analytes = st.multiselect("Select Analytes", analytes)

                # Step 4: Units selection
                units = st.selectbox(
                    "Select Units for Analytes",
                    options=units_list,
                    index=0
                )

                # Step 5: Select Confidence Interval via Slider
                confidence_level = st.slider(
                    "Select Confidence Level (%)",
                    min_value=80,
                    max_value=99,
                    value=95,
                    step=1
                )
                alpha = 1 - confidence_level / 100

                # Step 5b: Outlier exclusion option
                exclude_outliers = st.checkbox(
                    "Exclude outliers (>3 SD from mean difference)",
                    value=False,
                    help="Remove data points where the difference between methods is >3 standard deviations from the mean difference"
                )

                # Step 6: Run Deming Regression for selected analytes
                all_results = []
                for selected_analyte in selected_analytes:
                    result = deming_regression_analysis(df, analyzer_1, analyzer_2, selected_material, units, selected_analyte, confidence_level, alpha, exclude_outliers)
                    if result:
                        all_results.extend(result)

                # Step 7: Display results in combined summary table
                if all_results:
                    results_df = pd.DataFrame(all_results)
                    st.markdown("### 📊 Deming Regression Statistical Summary ")

                    # Show full results for all analytes and materials
                    st.dataframe(results_df)

                    
def deming_regression_analysis(df, analyzer_1, analyzer_2, selected_material, units, selected_analyte, confidence_level, alpha, exclude_outliers=False):
    # Filter the dataframe to include only the selected analyte and relevant columns
    ignore_cols = ['Material', 'Analyser', 'Date', 'Sample ID']
    
    # Ensure the analyte exists in the columns
    if selected_analyte not in df.columns:
        st.error(f"❌ {selected_analyte} column not found.")
        return None

    results_list = []

    # Subset the dataframe for the selected material and analyzers
    sub_df = df[(df['Material'] == selected_material) & df['Analyser'].isin([analyzer_1, analyzer_2])].copy()
    if sub_df.empty:
        st.warning(f"⚠ No data available for {selected_material} with the selected analyzers.")
        return None

    sub_df[selected_analyte] = pd.to_numeric(sub_df[selected_analyte], errors='coerce')
    pivot = sub_df.pivot_table(index='Sample ID', columns='Analyser', values=selected_analyte, aggfunc='mean')

    if analyzer_1 not in pivot or analyzer_2 not in pivot:
        st.warning("⚠ Data for both analyzers is missing.")
        return None

    pivot = pivot.dropna(subset=[analyzer_1, analyzer_2])
    if len(pivot) < 2:
        st.warning(f"⚠ Not enough data for {selected_analyte}. Skipping...")
        return None

    x, y = pivot[analyzer_1].values, pivot[analyzer_2].values
    n_original = len(x)
    
    # Outlier detection and removal if requested
    outliers_removed = 0
    if exclude_outliers and len(x) > 3:  # Need at least 3 points for meaningful outlier detection
        # Calculate differences between methods
        differences = y - x
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        # Identify outliers (>3 SD from mean difference)
        outlier_mask = np.abs(differences - mean_diff) > 3 * std_diff
        outliers_removed = np.sum(outlier_mask)
        
        if outliers_removed > 0:
            # Remove outliers
            x = x[~outlier_mask]
            y = y[~outlier_mask]
            
            # Check if we still have enough data
            if len(x) < 2:
                st.warning(f"⚠ Too many outliers removed for {selected_analyte}. Only {len(x)} points remaining.")
                return None
                
            st.info(f"📊 {outliers_removed} outlier(s) removed from {selected_analyte} analysis (>{3}SD from mean difference)")
    
    if len(x) < 2:
        st.warning(f"⚠ Not enough data for {selected_analyte}. Skipping...")
        return None

    def linear(B, x): return B[0] * x + B[1]
    model = Model(linear)
    odr_data = RealData(x, y)
    odr = ODR(odr_data, model, beta0=[1, 0])
    output = odr.run()

    slope, intercept = output.beta
    # USE CORRECT STANDARD ERRORS FROM ODR OUTPUT
    se_slope, se_intercept = output.sd_beta
    
    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # Degrees of freedom for Deming regression
    dof = len(x) - 2
    
    # t-value for selected confidence interval
    t_val = stats.t.ppf(1 - alpha / 2, dof)
    
    # Confidence intervals
    ci_slope = t_val * se_slope
    ci_intercept = t_val * se_intercept
    
    # Statistical test: H0: slope = 1 vs H1: slope ≠ 1
    t_stat = (slope - 1) / se_slope
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
    
    # Consistent outcome determination
    slope_lower = slope - ci_slope
    slope_upper = slope + ci_slope
    
    # Check if 1 is within confidence interval
    slope_ci_contains_1 = slope_lower <= 1 <= slope_upper
    
    # Determine outcome consistently
    if p_val == 0.0:
        outcome = ""  # Don't print outcome when p-value is exactly 0.0
    elif p_val <= 0.05:
        outcome = "Statistically significant bias"
    else:
        outcome = "No statistically significant bias"
    
    # Additional check - should be consistent with p-value (only if p_val != 0.0)
    if p_val != 0.0:
        if slope_ci_contains_1 and p_val > 0.05:
            outcome = "No statistically significant bias"
        elif not slope_ci_contains_1 and p_val <= 0.05:
            outcome = "Statistically significant bias"
        else:
            # If inconsistent, trust the p-value
            outcome = "Statistically significant bias" if p_val <= 0.05 else "No statistically significant bias"

    results_list.append({
        'Analyte': selected_analyte,
        'Material': selected_material,
        'Analyzer 1': analyzer_1,
        'Analyzer 2': analyzer_2,
        'Slope': round(slope, 3),
        'Intercept': round(intercept, 3),
        'R²': round(r_squared, 3),
        'n': len(x),
        'n_original': n_original,
        'Outliers_removed': outliers_removed,
        'Critical t-value': round(t_val, 3),
        'SE (Slope)': round(se_slope, 3),
        'CI Lower (Slope)': round(slope_lower, 3),
        'CI Upper (Slope)': round(slope_upper, 3),
        'p-value': round(p_val, 3),
        'Outcome': outcome
    })
    
    line_name = f"y = {slope:.2f}x + {intercept:.2f} (R² = {r_squared:.4f})"
    fig = go.Figure()

    # Scatter plot of the original data points
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Samples',
        marker=dict(color='steelblue'),
        hovertemplate=f"{analyzer_1}: %{{x:.2f}} {units}<br>{analyzer_2}: %{{y:.2f}} {units}<extra></extra>"
    ))

    # Regression line
    x_line = np.linspace(min(x), max(x), 100)
    y_line = slope * x_line + intercept
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name=line_name,
        line=dict(color='red')
    ))

    # CORRECT confidence interval calculation for regression line
    # This is an approximation - exact calculation is more complex
    x_mean = np.mean(x)
    se_pred = np.sqrt(se_intercept**2 + (x_line - x_mean)**2 * se_slope**2)
    
    y_upper = y_line + t_val * se_pred
    y_lower = y_line - t_val * se_pred
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_line, x_line[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill='toself',
        fillcolor=f'rgba(255, 99, 71, 0.3)',
        line=dict(color='rgba(255, 99, 71, 0)'),
        name=f'{confidence_level}% Confidence Interval',
        showlegend=True
    ))

    # Layout for the plot
    title_text = f"Deming Regression for {selected_analyte} ({selected_material})<br>"
    title_text += f"Slope: {slope:.3f} [{slope_lower:.3f}, {slope_upper:.3f}] | "
    title_text += f"p-value: {p_val:.3f}"
    
    # Add sample size info
    if outliers_removed > 0:
        title_text += f" | n={len(x)} ({outliers_removed} outliers removed)"
    else:
        title_text += f" | n={len(x)}"
    
    # Only add outcome to title if p_val is not 0.0
    if p_val != 0.0:
        title_text += f" | {outcome}"
    
    fig.update_layout(
        title=title_text,
        xaxis_title=f"{analyzer_1} ({units})",
        yaxis_title=f"{analyzer_2} ({units})",
        height=500,
        plot_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)

    return results_list