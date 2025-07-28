import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.odr import ODR, RealData, Model
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import apply_app_styling, units_list

# === Utility Functions (Keep these as they are from the previous good version) ===
def grubbs_test(values, alpha=0.05):
    """
    Original Grubbs test for outlier detection (single outlier detection)
    KEEPING THIS FOR BACKWARD COMPATIBILITY
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

def grubbs_test_iterative(values, alpha=0.05, max_iterations=10):
    """
    Perform iterative Grubbs test for outlier detection.
    This will continue removing outliers until no more are found or max_iterations is reached.
    """
    values = pd.Series(values).copy()
    outlier_indices = []
    original_indices = values.index.tolist()
    
    for iteration in range(max_iterations):
        n = len(values)
        if n < 3:  # Need at least 3 points for Grubbs test
            break
            
        # Calculate Grubbs statistic
        mean_val = values.mean()
        std_val = values.std(ddof=1)
        
        if std_val == 0:  # All values are the same
            break
            
        abs_diff = abs(values - mean_val)
        max_diff_idx = abs_diff.idxmax()
        G = abs_diff[max_diff_idx] / std_val
        
        # Critical value from Grubbs test table (two-sided)
        t_crit = stats.t.ppf(1 - alpha / (2 * n), df=n - 2)
        G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        
        if G > G_crit:
            # Found an outlier
            outlier_indices.append(max_diff_idx)
            values = values.drop(max_diff_idx)
        else:
            # No more outliers found
            break
    
    # Create boolean mask for original data
    is_outlier = np.array([idx in outlier_indices for idx in original_indices])
    return is_outlier

def identify_outliers(differences, alpha=0.05, method='grubbs_iterative', x_vals=None, y_vals=None):
    """
    Identify outliers using multiple methods and return a boolean array.
    This function is designed to be called by perform_outlier_detection_with_options.
    
    Parameters:
    - differences: array of difference values
    - alpha: significance level for Grubbs test
    - method: 'grubbs_iterative', 'grubbs_single', 'limits_only', 'large_percent_diff', or 'combined'
    - x_vals, y_vals: required for 'large_percent_diff' and 'combined' methods
    """
    if method == 'grubbs_iterative':
        return grubbs_test_iterative(differences, alpha)
    
    elif method == 'grubbs_single':
        return grubbs_test(differences, alpha)
    
    elif method == 'limits_only':
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
        return (differences > loa_upper) | (differences < loa_lower)
    
    elif method == 'large_percent_diff':
        if x_vals is None or y_vals is None:
            raise ValueError("x_vals and y_vals are required for 'large_percent_diff' method.")
        means = (x_vals + y_vals) / 2
        percent_diffs = np.abs(differences / means) * 100
        return percent_diffs > 50
    
    elif method == 'combined':
        # Combine Grubbs iterative with limit-based detection
        grubbs_outliers = grubbs_test_iterative(differences, alpha)
        
        if grubbs_outliers.any():
            clean_diffs = differences[~grubbs_outliers]
            if len(clean_diffs) > 0:
                clean_mean = np.mean(clean_diffs)
                clean_std = np.std(clean_diffs, ddof=1)
                clean_upper = clean_mean + 1.96 * clean_std
                clean_lower = clean_mean - 1.96 * clean_std
                
                limit_outliers = (differences > clean_upper) | (differences < clean_lower)
                return grubbs_outliers | limit_outliers
        
        return grubbs_outliers
    
    elif method == 'comprehensive': # This combines Grubbs Iterative and Large % Difference
        grubbs_iterative_outliers = grubbs_test_iterative(differences, alpha)
        
        if x_vals is None or y_vals is None:
            st.warning("x_vals and y_vals are required for 'Comprehensive' method to calculate percentage differences. Proceeding with Grubbs (Iterative) only.")
            return grubbs_iterative_outliers
            
        means = (x_vals + y_vals) / 2
        percent_diffs = np.abs(differences / means) * 100
        large_percent_outliers = percent_diffs > 50
        
        return grubbs_iterative_outliers | large_percent_outliers

    else:
        raise ValueError("Invalid outlier detection method.")


# Modified perform_outlier_detection_with_options to just apply the chosen method
# and display detailed outlier info, no longer selecting the method via UI
def get_outlier_analysis_results(merged_data, selected_analyte, analyzer_1, analyzer_2, alpha, chosen_method):
    """
    Performs outlier detection based on a pre-chosen method and returns the outlier flags
    along with a DataFrame of outlier details.
    
    Args:
        merged_data (pd.DataFrame): DataFrame containing Sample ID and paired analyzer values.
        selected_analyte (str): The name of the analyte.
        analyzer_1 (str): Name of the first analyzer.
        analyzer_2 (str): Name of the second analyzer.
        alpha (float): Significance level for statistical tests.
        chosen_method (str): The specific outlier detection method to use.

    Returns:
        tuple: (outlier_flags_boolean_array, outlier_details_dataframe)
    """
    vals1 = merged_data[f'{selected_analyte}_1']
    vals2 = merged_data[f'{selected_analyte}_2']
    diffs = vals1 - vals2

    # Identify outliers using the chosen method
    is_outlier = identify_outliers(diffs.values, alpha, chosen_method, x_vals=vals1.values, y_vals=vals2.values)
    
    outlier_details_df = pd.DataFrame()
    if np.sum(is_outlier) > 0:
        outlier_indices = np.where(is_outlier)[0]
        outlier_sample_ids = merged_data['Sample ID'].iloc[outlier_indices].tolist()
        
        outlier_details = []
        for i, idx in enumerate(outlier_indices):
            val1 = vals1.iloc[idx]
            val2 = vals2.iloc[idx]
            diff = diffs.iloc[idx]
            mean_val = (val1 + val2) / 2
            percent_diff = abs(diff / mean_val) * 100 if mean_val != 0 else 0
            
            # Recalculate mean/std of *all* initial differences to get z-score relative to original data
            initial_mean_diff = np.mean(diffs)
            initial_std_diff = np.std(diffs, ddof=1)
            z_score = abs(diff - initial_mean_diff) / initial_std_diff if initial_std_diff != 0 else np.nan

            outlier_details.append({
                'Sample ID': outlier_sample_ids[i],
                f'{analyzer_1}': round(val1, 3),
                f'{analyzer_2}': round(val2, 3),
                'Difference': round(diff, 3),
                'Mean': round(mean_val, 3),
                '% Difference': round(percent_diff, 1),
                'Z-Score (vs. Original)': round(z_score, 2) if not np.isnan(z_score) else "N/A",
                'Outside ±1.96SD (Original Data)': '✓' if not np.isnan(z_score) and abs(z_score) > 1.96 else '✗'
            })
        
        outlier_details_df = pd.DataFrame(outlier_details)
        
    return is_outlier, outlier_details_df


def run():
    apply_app_styling()

    st.title("⚖️ Deming Regression Analysis")

    st.markdown("---")
    st.markdown("### 📘 Information & Instructions")
    with st.expander("Deming Regression Explained"):
        st.markdown("""
        **Deming regression** is used when **both X and Y variables have measurement error**,
        which is common in method comparison studies.
        - **Slope**: Indicates proportional bias (ideal value is 1).
        - **Intercept**: Indicates constant bias (ideal value is 0).
        - **R²**: Represents the strength of the linear relationship between the two methods.
        """)

    with st.expander("How to Use This App"):
        st.markdown("""
        1.  **Upload your CSV file**: Ensure it contains 'Material', 'Analyser', 'Sample ID', and your analyte columns.
        2.  **Select Analyzers**: Choose the two analyzers you want to compare.
        3.  **Filter Data**: Select the material type and units relevant to your analysis.
        4.  **Choose Analytes**: Select one or more analytes to perform regression on.
        5.  **Configure Outlier Detection**: Decide whether to identify and/or exclude outliers, and choose your preferred method.
        6.  **Run Analysis**: Click the button to generate Deming regression plots and statistical summaries.
        """)
    st.markdown("---")

    # --- Start of the modified file upload section ---
    with st.expander("📤 Upload CSV File", expanded=True):
        uploaded_file = st.file_uploader(" ", type=["csv"], key="uploader")

        # Initialize session state variable for DataFrame
        if 'df' not in st.session_state:
            st.session_state.df = None
            st.session_state.uploaded_file_id = None # Initialize file ID as well

        if uploaded_file is not None:
            try:
                # Read the CSV only if a new file is uploaded or if df is not yet in session state
                # and a file is currently uploaded.
                # This prevents re-reading on every rerun if the file hasn't changed.
                if st.session_state.df is None or st.session_state.uploaded_file_id != uploaded_file.file_id:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.df = df
                    st.session_state.uploaded_file_id = uploaded_file.file_id # Store file ID to detect changes
                    
                    required_cols = ['Analyser', 'Material', 'Sample ID']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Missing required columns in your CSV: **{', '.join(missing_cols)}**. Please upload a file with these columns.")
                        st.session_state.df = None # Invalidate DF if essential columns are missing
                        st.stop() # Use st.stop() to halt execution more gracefully than return
                    
                    st.success(f"✅ File **'{uploaded_file.name}'** uploaded successfully!")
                    
                else:
                    st.info(f"Using previously uploaded file: **'{uploaded_file.name}'**")

            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}. Please ensure it's a valid CSV.")
                st.session_state.df = None # Clear df in case of error
                st.stop() # Use st.stop() to halt execution more gracefully
        elif st.session_state.df is None:
             st.info("Please upload a CSV file to begin your analysis.")
             st.stop() # Stop if no file is uploaded and df is None
    # --- End of the modified file upload section ---

    df = st.session_state.df # Use the DataFrame from session state

    if df is not None:
        st.markdown("---")
        st.markdown("### ⚙️ Analysis Settings")

        # Data Preprocessing: Ensure necessary columns exist (this check is already done above, but good to keep if user flow changes)
        required_cols = ['Analyser', 'Material', 'Sample ID']
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Critical error: Required columns {', '.join(required_cols)} are missing from the DataFrame. Please re-upload a valid CSV.")
            st.session_state.df = None # Force re-upload
            st.stop()

        # Step 1: Analyzer selection
        analyzers = df['Analyser'].dropna().unique()
        if len(analyzers) < 2:
            st.error("❌ Your data must contain results from at least two different analyzers for comparison.")
            st.stop()
        
        col1, col2 = st.columns(2)
        with col1:
            analyzer_1 = st.selectbox("Select Analyzer 1 (X-axis)", analyzers, index=0)
        with col2:
            analyzer_2 = st.selectbox("Select Analyzer 2 (Y-axis)", analyzers, 
                                      index=1 if len(analyzers) > 1 else 0)

        if analyzer_1 == analyzer_2:
            st.warning("⚠ Please select two different analyzers for comparison.")
            st.stop()

        # Step 2: Material selection
        valid_materials = ["EQA", "Patient"] # Define expected material types
        material_options = df['Material'].dropna().unique()
        filtered_materials = [m for m in material_options if m in valid_materials]
        
        if not filtered_materials:
            st.warning("⚠ No valid 'Material' types (EQA, Patient) found in your data. Please check your CSV.")
            st.stop()
        
        selected_material = st.selectbox(
            "Select Material Type",
            options=filtered_materials,
            index=0
        )

        # Step 3: Analyte selection
        # Exclude known metadata columns to get potential analyte columns
        metadata_cols = ['Date', 'Test', 'Material', 'Analyser', 'Sample ID', 'Batch ID', 'Lot Number']
        analytes = [col for col in df.columns if col not in metadata_cols]
        if not analytes:
            st.error("❌ No analyte columns found. Please ensure your CSV has numeric columns for analytes beyond metadata.")
            st.stop()

        selected_analytes = st.multiselect("Select Analytes for Analysis", analytes)
        if not selected_analytes:
            st.info("Please select at least one analyte to proceed with the analysis.")
            # We don't stop here, allow other settings to be chosen, then check again at "Run" button

        # Step 4: Units selection
        selected_units = st.selectbox(
            "🔎 Select Units for Analytes",
            options=units_list,
            index=0 # Default to the first unit
        )

        # Step 5: Confidence Interval via Slider
        confidence_level = st.slider(
            "😎 Select Confidence Level (%)",
            min_value=80,
            max_value=99,
            value=95,
            step=1
        )
        alpha = 1 - confidence_level / 100

        # --- Outlier detection and exclusion options (NOW TOGETHER BEFORE RUN BUTTON) ---
        st.markdown("---")
        st.markdown("### ⚙️ Outlier Management")
        use_outlier_detection = st.checkbox("🔍 Enable Outlier Identification", value=True)
        
        outlier_method_options_display = {
            "Grubbs (Single)": 'grubbs_single',
            "Grubbs (Iterative)": 'grubbs_iterative',
            "Limits Only (±1.96σ)": 'limits_only',
            "Large % Difference (>50%)": 'large_percent_diff',
            "Combined Method (Grubbs Iterative + Limits)": 'combined',
            "Comprehensive (Grubbs Iterative + Large % Difference)": 'comprehensive'
        }
        
        chosen_outlier_method_key = "None" # Default if detection is off
        if use_outlier_detection:
            selected_outlier_method_name = st.selectbox(
                "Select Outlier Detection Method:",
                options=list(outlier_method_options_display.keys()),
                index=5, # Default to "Comprehensive"
                help="""
                - **Grubbs (Single)**: Detects only the most extreme outlier.
                - **Grubbs (Iterative)**: Repeatedly applies Grubbs test until no more outliers are found.
                - **Limits Only (±1.96σ)**: Flags points outside ±1.96 standard deviations from the mean of differences.
                - **Large % Difference (>50%)**: Flags points where the absolute percentage difference is greater than 50%.
                - **Combined Method (Grubbs Iterative + Limits)**: Combines iterative Grubbs with checking against recalculated ±1.96σ limits after Grubbs outliers are notionally removed.
                - **Comprehensive (Grubbs Iterative + Large % Difference)**: Combines iterative Grubbs with flagging points based on a large percentage difference (RECOMMENDED).
                """
            )
            # Map display name to internal method key
            chosen_outlier_method_key = outlier_method_options_display[selected_outlier_method_name]
        else:
            selected_outlier_method_name = "Disabled" # For display in results

        exclude_outliers = st.checkbox(
            "❌ Exclude detected outliers from regression calculation", 
            value=False,
            help="If checked, identified outliers will not be used in the Deming regression calculation. They will still be shown as red squares on the plot."
        )
        
        # Step 6: Run Analysis Button
        st.markdown("---")
        if st.button("🚀 Run Deming Regression Analysis", type="primary"):
            if not selected_analytes:
                st.error("❌ Please select at least one analyte to analyze.")
                st.stop() # Stop here if no analytes are selected

            all_results = []
            # This loop will now execute only when the button is pressed
            for selected_analyte in selected_analytes:
                st.markdown(f"## Results for {selected_analyte} ({selected_units})")
                
                # --- Data Preparation for a single analyte ---
                sub_df = df[(df['Material'] == selected_material) & 
                            df['Analyser'].isin([analyzer_1, analyzer_2])].copy()
                
                if sub_df.empty:
                    st.warning(f"⚠ No data available for {selected_material} with the selected analyzers for {selected_analyte}.")
                    continue

                sub_df[selected_analyte] = pd.to_numeric(sub_df[selected_analyte], errors='coerce')
                pivot = sub_df.pivot_table(index='Sample ID', columns='Analyser', values=selected_analyte, aggfunc='mean')

                if analyzer_1 not in pivot or analyzer_2 not in pivot:
                    st.warning(f"⚠ Data for both selected analyzers ({analyzer_1}, {analyzer_2}) is missing for {selected_analyte}. Skipping.")
                    continue

                pivot = pivot.dropna(subset=[analyzer_1, analyzer_2])
                if len(pivot) < 2:
                    st.warning(f"⚠ Not enough paired data points for {selected_analyte} after filtering. Need at least 2 points.")
                    continue

                original_x = pivot[analyzer_1].values
                original_y = pivot[analyzer_2].values
                original_sample_ids = pivot.index.tolist()
                n_original = len(original_x)

                x = original_x.copy()
                y = original_y.copy()
                sample_ids = original_sample_ids.copy()

                outlier_flags = np.array([False] * n_original)
                outlier_details_df = pd.DataFrame() # Initialize empty DataFrame
                outliers_removed_count = 0

                if use_outlier_detection and chosen_outlier_method_key != "None" and n_original >= 3:
                    merged_data_for_outliers = pd.DataFrame({
                        'Sample ID': original_sample_ids,
                        f'{selected_analyte}_1': original_x,
                        f'{selected_analyte}_2': original_y
                    })
                    
                    # Call the function to get outlier flags and details based on the pre-chosen method
                    outlier_flags, outlier_details_df = get_outlier_analysis_results(
                        merged_data_for_outliers, selected_analyte, analyzer_1, analyzer_2, alpha, chosen_outlier_method_key
                    )
                    
                    st.markdown(f"**Outlier Identification for {selected_analyte} using '{selected_outlier_method_name}':**")
                    if np.sum(outlier_flags) == 0:
                        st.success(f"✅ No outliers detected for {selected_analyte}.")
                    else:
                        st.info(f"💡 {np.sum(outlier_flags)} outlier(s) detected for {selected_analyte}.")
                        st.dataframe(outlier_details_df, use_container_width=True, hide_index=True)

                    if exclude_outliers and np.sum(outlier_flags) > 0:
                        x = original_x[~outlier_flags]
                        y = original_y[~outlier_flags]
                        sample_ids = [original_sample_ids[i] for i in np.where(~outlier_flags)[0]]
                        outliers_removed_count = np.sum(outlier_flags)
                        
                        if len(x) < 2:
                            st.error(f"❌ After excluding {outliers_removed_count} outlier(s), not enough data points remain for {selected_analyte} to perform regression. Skipping.")
                            continue
                elif use_outlier_detection and n_original < 3:
                     st.warning(f"ℹ️ Not enough data points ({n_original}) for {selected_analyte} to perform outlier detection. Skipping outlier analysis.")
                     
                # Set outlier method applied string for results table
                outlier_method_applied_for_results = selected_outlier_method_name


                # --- Perform Deming Regression ---
                if len(x) < 2:
                    st.warning(f"⚠ Not enough data for {selected_analyte} after processing. Skipping Deming regression.")
                    continue

                def linear(B, x_data): 
                    return B[0] * x_data + B[1]
                
                model = Model(linear)
                odr_data = RealData(x, y)
                odr = ODR(odr_data, model, beta0=[1, 0])
                output = odr.run()
                
                slope, intercept = output.beta
                se_slope, se_intercept = output.sd_beta
                
                # Calculate R-squared for the regression on the *used* data
                y_pred_reg = slope * x + intercept
                ss_res_reg = np.sum((y - y_pred_reg) ** 2)
                ss_tot_reg = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res_reg / ss_tot_reg) if ss_tot_reg != 0 else np.nan
                
                # Statistical tests
                dof = len(x) - 2
                if dof <= 0:
                    st.warning(f"Insufficient degrees of freedom for statistical tests for {selected_analyte}. Skipping p-value and CI calculation.")
                    t_val, ci_slope, ci_intercept, t_stat, p_val_slope, slope_lower, slope_upper, outcome = [np.nan] * 8
                else:
                    t_val = stats.t.ppf(1 - alpha / 2, dof)
                    ci_slope = t_val * se_slope
                    ci_intercept = t_val * se_intercept
                    
                    # Test if slope is significantly different from 1
                    t_stat_slope = (slope - 1) / se_slope
                    p_val_slope = 2 * (1 - stats.t.cdf(abs(t_stat_slope), dof))

                    slope_lower = slope - ci_slope
                    slope_upper = slope + ci_slope
                    
                    outcome = "No statistically significant proportional bias"
                    if not (slope_lower <= 1 <= slope_upper):
                        outcome = "Statistically significant proportional bias"
                
                # --- Jackknife Calculations ---
                jk_slopes = []
                jk_intercepts = []
                n_used = len(x)
                
                if n_used >= 2: # Need at least 2 points to perform regression
                    for i in range(n_used):
                        # Leave one out
                        x_jk = np.delete(x, i)
                        y_jk = np.delete(y, i)

                        if len(x_jk) < 2: # Ensure enough points remain for regression
                            jk_slopes.append(np.nan)
                            jk_intercepts.append(np.nan)
                            continue

                        # Perform Deming regression on the leave-one-out subset
                        try:
                            odr_data_jk = RealData(x_jk, y_jk)
                            odr_jk = ODR(odr_data_jk, model, beta0=[1, 0])
                            output_jk = odr_jk.run()
                            jk_slopes.append(output_jk.beta[0])
                            jk_intercepts.append(output_jk.beta[1])
                        except Exception as e:
                            st.warning(f"Jackknife regression failed for sample {i} of {selected_analyte}: {e}")
                            jk_slopes.append(np.nan)
                            jk_intercepts.append(np.nan)
                    
                    jk_slopes_clean = np.array([s for s in jk_slopes if not np.isnan(s)])
                    jk_intercepts_clean = np.array([i_val for i_val in jk_intercepts if not np.isnan(i_val)])

                    if len(jk_slopes_clean) > 0:
                        # Calculate pseudo-values
                        pseudo_slopes = n_used * slope - (n_used - 1) * jk_slopes_clean
                        pseudo_intercepts = n_used * intercept - (n_used - 1) * jk_intercepts_clean

                        # Jackknife estimates
                        jk_slope_estimate = np.mean(pseudo_slopes)
                        jk_intercept_estimate = np.mean(pseudo_intercepts)

                        # Jackknife standard errors
                        jk_se_slope = np.sqrt(np.sum((pseudo_slopes - jk_slope_estimate)**2) / (n_used * (n_used - 1))) if n_used > 1 else np.nan
                        jk_se_intercept = np.sqrt(np.sum((pseudo_intercepts - jk_intercept_estimate)**2) / (n_used * (n_used - 1))) if n_used > 1 else np.nan

                        # Jackknife bias
                        jk_bias_slope = (n_used - 1) * (jk_slope_estimate - slope)
                        jk_bias_intercept = (n_used - 1) * (jk_intercept_estimate - intercept)

                    else: # No valid jackknife calculations
                        jk_slope_estimate, jk_intercept_estimate = np.nan, np.nan
                        jk_se_slope, jk_se_intercept = np.nan, np.nan
                        jk_bias_slope, jk_bias_intercept = np.nan, np.nan
                        st.warning(f"Not enough valid Jackknife calculations for {selected_analyte}. Results will be N/A.")

                else: # Not enough points for Jackknife
                    jk_slope_estimate, jk_intercept_estimate = np.nan, np.nan
                    jk_se_slope, jk_se_intercept = np.nan, np.nan
                    jk_bias_slope, jk_bias_intercept = np.nan, np.nan
                    st.warning(f"Not enough data points ({n_used}) for {selected_analyte} to perform Jackknife resampling. Skipping Jackknife calculations.")
                
                results_list_entry = {
                    'Analyte': selected_analyte,
                    'Material': selected_material,
                    'Analyzer 1 (X)': analyzer_1,
                    'Analyzer 2 (Y)': analyzer_2,
                    'Slope': round(slope, 3),
                    'Intercept': round(intercept, 3),
                    'R² (on used data)': round(r_squared, 3),
                    'N (Original)': n_original,
                    'N (Used in Regression)': len(x),
                    'Outliers Detected': np.sum(outlier_flags),
                    'Outliers Excluded': outliers_removed_count,
                    'Outlier Method Used': outlier_method_applied_for_results, # Use the string for results table
                    f'{confidence_level}% CI Lower (Slope)': round(slope_lower, 3),
                    f'{confidence_level}% CI Upper (Slope)': round(slope_upper, 3),
                    'p-value (Slope vs. 1)': round(p_val_slope, 4) if not np.isnan(p_val_slope) else "N/A",
                    'Outcome (Proportional Bias)': outcome,
                    'Jackknife Slope Estimate': round(jk_slope_estimate, 3) if not np.isnan(jk_slope_estimate) else "N/A",
                    'Jackknife Intercept Estimate': round(jk_intercept_estimate, 3) if not np.isnan(jk_intercept_estimate) else "N/A",
                    'Jackknife SE (Slope)': round(jk_se_slope, 3) if not np.isnan(jk_se_slope) else "N/A",
                    'Jackknife SE (Intercept)': round(jk_se_intercept, 3) if not np.isnan(jk_se_intercept) else "N/A",
                    'Jackknife Bias (Slope)': round(jk_bias_slope, 3) if not np.isnan(jk_bias_slope) else "N/A",
                    'Jackknife Bias (Intercept)': round(jk_bias_intercept, 3) if not np.isnan(jk_bias_intercept) else "N/A"
                }
                all_results.append(results_list_entry)

                # --- Plotting ---
                st.markdown(f"### Regression Plot for {selected_analyte}")
                fig = go.Figure()

                # Plot regular samples (those used in regression)
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    name='Samples Used in Regression',
                    marker=dict(color='steelblue', size=8),
                    customdata=np.array(sample_ids)[:, np.newaxis], # For hovertemplate
                    hovertemplate=f"Sample ID: %{{customdata[0]}}<br>{analyzer_1}: %{{x:.2f}} {selected_units}<br>{analyzer_2}: %{{y:.2f}} {selected_units}<extra></extra>"
                ))
                
                # Plot outliers (from original data) as red squares
                if np.sum(outlier_flags) > 0:
                    outlier_x_orig = original_x[outlier_flags]
                    outlier_y_orig = original_y[outlier_flags]
                    outlier_sample_ids_orig = [original_sample_ids[i] for i in np.where(outlier_flags)[0]]

                    outlier_status_text = "Excluded" if exclude_outliers else "Included"
                    fig.add_trace(go.Scatter(
                        x=outlier_x_orig,
                        y=outlier_y_orig,
                        mode='markers',
                        name=f'Outliers ({outlier_status_text})',
                        marker=dict(symbol='square', color='red', size=9, line=dict(width=1, color='DarkRed')),
                        customdata=np.array(outlier_sample_ids_orig)[:, np.newaxis],
                        hovertemplate=f"OUTLIER ({outlier_status_text.upper()})<br>Sample ID: %{{customdata[0]}}<br>{analyzer_1}: %{{x:.2f}} {selected_units}<br>{analyzer_2}: %{{y:.2f}} {selected_units}<extra></extra>"
                    ))

                # Regression line
                x_line = np.linspace(min(x) - 0.1 * (max(x) - min(x)), max(x) + 0.1 * (max(x) - min(x)), 100)
                y_line = slope * x_line + intercept

                # Confidence bands for the regression line
                n_bootstrap = 500 # Reduced for performance, increase for accuracy
                y_bootstrap = np.zeros((n_bootstrap, len(x_line)))
                
                for i in range(n_bootstrap):
                    bootstrap_indices = np.random.choice(len(x), size=len(x), replace=True)
                    x_boot = x[bootstrap_indices]
                    y_boot = y[bootstrap_indices]
                    
                    try:
                        odr_data_boot = RealData(x_boot, y_boot)
                        odr_boot = ODR(odr_data_boot, model, beta0=[slope, intercept])
                        output_boot = odr_boot.run()
                        slope_boot, intercept_boot = output_boot.beta
                        y_bootstrap[i] = slope_boot * x_line + intercept_boot
                    except Exception as e:
                        # Fallback if bootstrap fit fails for a sample
                        y_bootstrap[i] = slope * x_line + intercept 
                
                alpha_bootstrap = 1 - confidence_level / 100
                y_lower_ci = np.percentile(y_bootstrap, 100 * alpha_bootstrap / 2, axis=0)
                y_upper_ci = np.percentile(y_bootstrap, 100 * (1 - alpha_bootstrap / 2), axis=0)

                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_upper_ci,
                    line=dict(width=0),
                    mode='lines',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                fig.add_trace(go.Scatter(
                    x=x_line,
                    y=y_lower_ci,
                    line=dict(width=0),
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(70,130,180,0.2)',
                    name=f"{confidence_level}% Confidence Interval",
                    hoverinfo='skip'
                ))

                # Identity line y = x
                min_val = min(min(original_x), min(original_y))
                max_val = max(max(original_x), max(original_y))
                identity_line = np.linspace(min_val * 0.9, max_val * 1.1, 100) # Extend to cover plot range

                fig.add_trace(go.Scatter(
                    x=identity_line,
                    y=identity_line,
                    mode='lines',
                    name='Identity Line (y = x)',
                    line=dict(color='red', width=2, dash='dash'),
                    showlegend=True
                ))

                fig.update_layout(
                    title=f"Deming Regression: {selected_analyte} ({selected_units})",
                    xaxis_title=f"{analyzer_1} ({selected_units})",
                    yaxis_title=f"{analyzer_2} ({selected_units})",
                    legend=dict(x=0.01, y=0.99),
                    template='plotly_white',
                    hovermode='closest'
                )
                st.plotly_chart(fig, use_container_width=True)

                # --- Residual Plot ---
                st.markdown(f"### Residual Plot for {selected_analyte}")

                # Calculate residuals for ALL original data points using the fitted regression line
                # This shows how even excluded outliers would fit the line
                deming_pred_all = intercept + slope * original_x
                deming_residuals_all = original_y - deming_pred_all

                fig_residuals = go.Figure()

                # Plot residuals for samples used in regression
                samples_used_mask = ~outlier_flags if exclude_outliers else np.array([True] * n_original)
                
                fig_residuals.add_trace(go.Scatter(
                    x=original_x[samples_used_mask],
                    y=deming_residuals_all[samples_used_mask],
                    mode='markers',
                    name='Residuals (Used in Regression)',
                    marker=dict(color='orange', size=8),
                    customdata=np.array(original_sample_ids)[samples_used_mask][:, np.newaxis],
                    hovertemplate=f"Sample ID: %{{customdata[0]}}<br>{analyzer_1}: %{{x:.2f}} {selected_units}<br>Residual: %{{y:.2f}} {selected_units}<extra></extra>"
                ))
                
                # Plot residuals for detected outliers (if any)
                if np.sum(outlier_flags) > 0:
                    outlier_x_resid = original_x[outlier_flags]
                    outlier_residuals = deming_residuals_all[outlier_flags]
                    outlier_sample_ids_resid = [original_sample_ids[i] for i in np.where(outlier_flags)[0]]

                    outlier_status_text = "Excluded" if exclude_outliers else "Included"
                    fig_residuals.add_trace(go.Scatter(
                        x=outlier_x_resid,
                        y=outlier_residuals,
                        mode='markers',
                        name=f'Outlier Residuals ({outlier_status_text})',
                        marker=dict(symbol='square', color='red', size=9, line=dict(width=1, color='DarkRed')),
                        customdata=np.array(outlier_sample_ids_resid)[:, np.newaxis],
                        hovertemplate=f"OUTLIER ({outlier_status_text.upper()})<br>Sample ID: %{{customdata[0]}}<br>{analyzer_1}: %{{x:.2f}} {selected_units}<br>Residual: %{{y:.2f}} {selected_units}<extra></extra>"
                    ))

                # Add a zero line to the residual plot
                fig_residuals.add_hline(y=0, line_dash="dash", line_color="grey", annotation_text="Zero Residual", 
                                        annotation_position="bottom right")

                fig_residuals.update_layout(
                    title=f"Residual Plot: {selected_analyte}",
                    xaxis_title=f"Initial Analyzer ({analyzer_1}) ({selected_units})",
                    yaxis_title=f"Residual ({analyzer_2} - Predicted {analyzer_2}) ({selected_units})",
                    template='plotly_white',
                    hovermode='closest'
                )
                st.plotly_chart(fig_residuals, use_container_width=True)
                st.markdown("---") # Separator between analyte results

            # Final Combined Summary Table
            if all_results:
                st.markdown("## ✨ Combined Deming Regression Summary")
                results_df = pd.DataFrame(all_results)
                st.dataframe(results_df, use_container_width=True)

                # Download button for combined results
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download All Results as CSV",
                    data=csv,
                    file_name=f"deming_regression_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
            else:
                st.error("No successful analyses to summarize. Please check your data and selections.")

# This ensures the app runs when the script is executed
if __name__ == "__main__":
    run()