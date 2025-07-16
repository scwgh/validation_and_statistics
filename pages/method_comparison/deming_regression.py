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
        with st.expander("⚙️ Settings: Selection and Outlier Detection", expanded=True):
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
                        "🔎 Select Units for Analytes",
                        options=units_list,
                        index=0
                    )

                    # Step 5: Select Confidence Interval via Slider
                    confidence_level = st.slider(
                        "😎 Select Confidence Level (%)",
                        min_value=80,
                        max_value=99,
                        value=95,
                        step=1
                    )
                    alpha = 1 - confidence_level / 100

                    # Step 5b: Outlier exclusion option
                    exclude_outliers = st.checkbox(
                        "❌ Exclude outliers (>3 SD from mean difference)",
                        value=False,
                        help="Remove data points where the difference between methods is >3 standard deviations from the mean difference"
                    )

        # Step 6: Run Analysis Button
        if st.button("🔍 Run Deming Regression Analysis", type="primary"):
            if not selected_analytes:
                st.error("❌ Please select at least one analyte to analyze.")
            else:
                # Step 7: Run Deming Regression for selected analytes
                all_results = []
                for selected_analyte in selected_analytes:
                    result = deming_regression_analysis(df, analyzer_1, analyzer_2, selected_material, units, selected_analyte, confidence_level, alpha, exclude_outliers)
                    if result:
                        all_results.extend(result)

                # Step 8: Display results in combined summary table
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
    se_slope, se_intercept = output.sd_beta
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    dof = len(x) - 2
    t_val = stats.t.ppf(1 - alpha / 2, dof)
    ci_slope = t_val * se_slope
    ci_intercept = t_val * se_intercept
    t_stat = (slope - 1) / se_slope
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
    
    slope_lower = slope - ci_slope
    slope_upper = slope + ci_slope
    
    slope_ci_contains_1 = slope_lower <= 1 <= slope_upper
    
    if p_val == 0.0:
        outcome = ""  
    elif p_val <= 0.05:
        outcome = "Statistically significant bias"
    else:
        outcome = "No statistically significant bias"
    if p_val != 0.0:
        if slope_ci_contains_1 and p_val > 0.05:
            outcome = "No statistically significant bias"
        elif not slope_ci_contains_1 and p_val <= 0.05:
            outcome = "Statistically significant bias"
        else:
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
    
    # Create plot section
    st.markdown(f"### 📈 Deming Regression Plot: {selected_analyte}")
    
    line_name = f"y = {slope:.2f}x + {intercept:.2f} (R² = {r_squared:.4f})"
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Samples',
        marker=dict(color='steelblue', size=8),
        hovertemplate=f"{analyzer_1}: %{{x:.2f}} {units}<br>{analyzer_2}: %{{y:.2f}} {units}<extra></extra>"
    ))

    x_line = np.linspace(min(x), max(x), 100)
    y_line = slope * x_line + intercept

    # FIXED: Proper confidence bands calculation for Deming regression
    # Get the covariance matrix from ODR output
    cov_matrix = output.cov_beta
    
    # Calculate confidence bands using bootstrap approach for better accuracy
    n_bootstrap = 1000
    y_bootstrap = np.zeros((n_bootstrap, len(x_line)))
    
    # Generate bootstrap samples
    for i in range(n_bootstrap):
        # Bootstrap sample indices
        bootstrap_indices = np.random.choice(len(x), size=len(x), replace=True)
        x_boot = x[bootstrap_indices]
        y_boot = y[bootstrap_indices]
        
        # Fit Deming regression on bootstrap sample
        try:
            odr_data_boot = RealData(x_boot, y_boot)
            odr_boot = ODR(odr_data_boot, model, beta0=[slope, intercept])
            output_boot = odr_boot.run()
            slope_boot, intercept_boot = output_boot.beta
            y_bootstrap[i] = slope_boot * x_line + intercept_boot
        except:
            # If bootstrap fails, use original estimates
            y_bootstrap[i] = slope * x_line + intercept
    
    # Calculate confidence intervals from bootstrap distribution
    alpha_bootstrap = 1 - confidence_level / 100
    y_lower = np.percentile(y_bootstrap, 100 * alpha_bootstrap / 2, axis=0)
    y_upper = np.percentile(y_bootstrap, 100 * (1 - alpha_bootstrap / 2), axis=0)

    # Alternative method using analytical approach if bootstrap bands are too narrow
    # Calculate prediction error using ODR covariance matrix
    if np.max(y_upper - y_lower) < 0.01 * np.max(y_line):  # If bands are too narrow
        # Use analytical approach with ODR covariance matrix
        var_slope = cov_matrix[0, 0] if cov_matrix is not None else se_slope**2
        var_intercept = cov_matrix[1, 1] if cov_matrix is not None else se_intercept**2
        covar_slope_intercept = cov_matrix[0, 1] if cov_matrix is not None else 0
        
        # Calculate variance of predicted y at each x
        var_y_pred = (x_line**2 * var_slope + 
                     var_intercept + 
                     2 * x_line * covar_slope_intercept)
        
        # Add residual variance for prediction intervals
        residual_var = np.sum((y - y_pred)**2) / (len(x) - 2)
        total_var = var_y_pred + residual_var
        
        # Calculate confidence bands
        margin_of_error = t_val * np.sqrt(total_var)
        y_upper = y_line + margin_of_error
        y_lower = y_line - margin_of_error

    # Add confidence bands FIRST (so they appear behind other lines)
    # Create the filled area by adding upper bound, then lower bound with fill
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_upper,
        line=dict(width=0),
        mode='lines',
        showlegend=False,
        hoverinfo='skip',
        name='Upper CI'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_lower,
        line=dict(width=0),
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(70,130,180,0.2)',
        name=f"{confidence_level}% CI",
        hoverinfo='skip'
    ))

    # Identity line y = x
    fig.add_trace(go.Scatter(
        x=x_line,
        y=x_line,
        mode='lines',
        name='Identity line (y = x)',
        line=dict(color='red', width=2, dash='dash'),
        showlegend=True
    ))

    # Regression line
    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name=line_name,
        line=dict(color='darkgreen', width=2),
        hoverinfo='skip'
    ))

    # Final layout
    fig.update_layout(
        title=f"Deming Regression: {selected_analyte}",
        xaxis_title=f"{analyzer_1} ({units})",
        yaxis_title=f"{analyzer_2} ({units})",
        legend=dict(x=0.01, y=0.99),
        template='plotly_white',
        width=800,
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- JACKKNIFE ESTIMATION FOR SLOPE & INTERCEPT ---
    slopes = []
    intercepts = []
    for i in range(len(x)):
        x_jack = np.delete(x, i)
        y_jack = np.delete(y, i)
        odr_data_jack = RealData(x_jack, y_jack)
        odr_jack = ODR(odr_data_jack, model, beta0=[1, 0])
        output_jack = odr_jack.run()
        slopes.append(output_jack.beta[0])
        intercepts.append(output_jack.beta[1])

    # Convert to numpy arrays
    slopes = np.array(slopes)
    intercepts = np.array(intercepts)

    # Mean Jackknife Estimates
    mean_slope = np.mean(slopes)
    mean_intercept = np.mean(intercepts)

    # Standard Errors
    se_slope_jack = np.sqrt((len(x) - 1) / len(x) * np.sum((slopes - mean_slope) ** 2))
    se_intercept_jack = np.sqrt((len(x) - 1) / len(x) * np.sum((intercepts - mean_intercept) ** 2))

    # Confidence intervals
    ci_jack_slope = t_val * se_slope_jack
    ci_jack_intercept = t_val * se_intercept_jack

    slope_lower_jack = mean_slope - ci_jack_slope
    slope_upper_jack = mean_slope + ci_jack_slope
    intercept_lower_jack = mean_intercept - ci_jack_intercept
    intercept_upper_jack = mean_intercept + ci_jack_intercept

    # Display jackknife results
    st.markdown(f"#### 🔁 Jackknife Estimates for {selected_analyte}")
    jack_df = pd.DataFrame({
        "Parameter": ["Slope", "Intercept"],
        "Mean Jackknife Estimate": [round(mean_slope, 3), round(mean_intercept, 3)],
        "Standard Error": [round(se_slope_jack, 3), round(se_intercept_jack, 3)],
        f"Lower {confidence_level}% CI": [round(slope_lower_jack, 3), round(intercept_lower_jack, 3)],
        f"Upper {confidence_level}% CI": [round(slope_upper_jack, 3), round(intercept_upper_jack, 3)],
    })
    st.dataframe(jack_df, use_container_width=True)
    
    st.markdown("---")  # Add separator between analytes
    
    return results_list

# Add references at the bottom
st.markdown("---")
st.subheader("📚 References")
st.markdown("""
**Westgard, J.O., Barry, P.L., and Hunt, M.R. (1981)**, *A Multi-Rule Shewhart Chart for Quality Control in Clinical Chemistry*, Clinical Chemistry, 27 (3), pp.493-501
(https://westgard.com/downloads/papers-downloads/27-westgard-rules-paper/file.html)
""")