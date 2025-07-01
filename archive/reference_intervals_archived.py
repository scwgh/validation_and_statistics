import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, shapiro, jarque_bera, anderson, gaussian_kde
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from utils import apply_app_styling
import warnings
warnings.filterwarnings('ignore')

# --- Page Setup ---
st.set_page_config(
    page_title="Reference Interval Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_styling()

st.markdown('<h1 class="main-header">📊 Reference Interval Analysis</h1>', unsafe_allow_html=True)

with st.expander("📘 What is Reference Interval Analysis?", expanded=True):
    st.markdown("""
        A **reference interval** defines the range of values expected for a healthy population.
        Typically, this is the central 95% of the data (2.5th to 97.5th percentile).

        This tool supports both:
        - **Non-parametric estimation** (using percentiles) - Recommended for most cases
        - **Parametric estimation** (assuming normally distributed data)
        
        **Minimum sample size**: 120 individuals (as per CLSI C28-A3 guidelines), but will calculate with fewer samples and show warnings.
        
        **Key Features**:
        - Automatic age calculation from dates
        - Multiple normality tests
        - Outlier detection and exclusion
        - Age-partitioned reference intervals
        - Interactive visualizations
        - Comprehensive quality checks
    """)

with st.expander("📘 Reference Interval Metrics Explained:", expanded=False):
    st.markdown("""
    - **Lower Limit**: The 2.5th percentile (or mean - 1.96 × SD for parametric)
    - **Upper Limit**: The 97.5th percentile (or mean + 1.96 × SD for parametric)
    - **N**: Number of valid observations used in the calculation
    - **Mean**: Average value of the analyte
    - **Median**: Middle value when data is sorted (robust central tendency)
    - **SD**: Standard deviation (measure of spread)
    - **CV%**: Coefficient of variation (SD/Mean × 100) - relative variability
    - **Skewness**: Measure of asymmetry (0 = symmetric, >0 = right-skewed, <0 = left-skewed)
    - **Kurtosis**: Measure of tail heaviness (3 = normal, >3 = heavy tails, <3 = light tails)
    """)

with st.expander("📘 Instructions:", expanded=False):
    st.markdown("""
    **To get started:**

    1. Upload your `.csv` file containing results from healthy individuals.
    2. **Required columns:**
        - `Date of Analysis`, `Batch ID`, `Sample ID`, `Date of Birth`
    3. **Optional columns** (will be calculated if missing):
        - `Age (y)`, `Age (m)`, `Age (w)` - Ages in years, months, weeks
        - `Gender` - For gender-specific analysis
    4. **Analyte columns** should appear after the demographic columns.
    5. Apply optional filters for gender, age, or specific batches.
    6. Choose the reference interval method (Non-parametric recommended).
    7. View distributions and download your calculated reference intervals.
    
    **Advanced Features**:
    - **Outlier Detection**: Exclude extreme values (>3SD from mean)
    - **Age Partitioning**: Create age-specific reference intervals
    - **Batch Analysis**: Check for batch effects
    - **Quality Metrics**: Multiple normality tests and distribution assessments
    
    **Note:** Ages will be automatically calculated from Date of Birth and Date of Analysis if not provided.
    All analyte columns will be automatically converted to numeric values.
    """)

def convert_to_numeric_enhanced(df, columns):
    conversion_summary = {}
    for col in columns:
        if col in df.columns:
            original_data = df[col].copy()
            original_count = len(original_data.dropna())
            
            original_non_null = original_data.dropna()
            
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            numeric_data = df[col].dropna()
            numeric_count = len(numeric_data)
            failed_conversions = original_count - numeric_count
            
            problematic_values = []
            if failed_conversions > 0:
                for idx, (orig, new) in enumerate(zip(original_non_null, pd.to_numeric(original_non_null, errors='coerce'))):
                    if pd.isna(new) and not pd.isna(orig):
                        problematic_values.append(str(orig))
                        if len(problematic_values) >= 5:  
                            break
            
            conversion_summary[col] = {
                'original_count': original_count,
                'numeric_count': numeric_count,
                'failed_conversions': failed_conversions,
                'success_rate': (numeric_count / original_count * 100) if original_count > 0 else 0,
                'problematic_values': problematic_values
            }
    
    return df, conversion_summary

def perform_comprehensive_normality_tests(data):
    clean_data = pd.to_numeric(data, errors='coerce').dropna()
    if len(clean_data) < 3:
        return None
    results = {'n': len(clean_data)}
    try:
        if len(clean_data) <= 5000:
            shapiro_stat, shapiro_p = shapiro(clean_data)
            results['shapiro_p'] = shapiro_p
            results['shapiro_normal'] = shapiro_p > 0.05
        
        if len(clean_data) >= 8:
            jb_stat, jb_p = jarque_bera(clean_data)
            results['jarque_bera_p'] = jb_p
            results['jarque_bera_normal'] = jb_p > 0.05
        
        # Anderson-Darling test
        if len(clean_data) >= 8:
            ad_result = anderson(clean_data, dist='norm')
            results['anderson_darling_stat'] = ad_result.statistic
            results['anderson_darling_critical'] = ad_result.critical_values[2]
            results['anderson_darling_normal'] = ad_result.statistic < ad_result.critical_values[2]
        
        # Calculate skewness and kurtosis
        from scipy.stats import skew, kurtosis
        results['skewness'] = skew(clean_data)
        results['kurtosis'] = kurtosis(clean_data, fisher=False)  # Pearson's kurtosis
        
        # Overall normality assessment
        normal_tests = [v for k, v in results.items() if k.endswith('_normal')]
        if normal_tests:
            results['overall_normal'] = sum(normal_tests) >= len(normal_tests) / 2
        
    except Exception as e:
        st.warning(f"Error in normality testing: {str(e)}")
        results['error'] = str(e)
    
    return results

def calculate_reference_interval_enhanced(data, method="non-parametric", confidence_level=0.95, 
                                        exclude_outliers=False, outlier_method="3sd"):

    clean_data = pd.to_numeric(data, errors='coerce').dropna()
    
    if len(clean_data) == 0:
        return None
    
    # Outlier detection and removal
    if exclude_outliers and len(clean_data) > 10:
        if outlier_method == "3sd":
            mean_val = clean_data.mean()
            std_val = clean_data.std()
            outlier_mask = np.abs(clean_data - mean_val) <= 3 * std_val
            clean_data = clean_data[outlier_mask]
        elif outlier_method == "iqr":
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (clean_data >= Q1 - 1.5 * IQR) & (clean_data <= Q3 + 1.5 * IQR)
            clean_data = clean_data[outlier_mask]
    
    # Calculate basic statistics
    n = len(clean_data)
    mean_val = clean_data.mean()
    median_val = clean_data.median()
    std_val = clean_data.std(ddof=1)  # Sample standard deviation
    cv_val = (std_val / mean_val) * 100 if mean_val != 0 else np.nan
    
    # Calculate percentiles for reference interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    if method.lower() == "non-parametric":
        # Use percentile method (recommended)
        if n >= 3:
            # Use different interpolation methods based on sample size
            interp_method = 'linear' if n >= 20 else 'nearest'
            lower_limit = np.percentile(clean_data, lower_percentile, method=interp_method)
            upper_limit = np.percentile(clean_data, upper_percentile, method=interp_method)
        else:
            # Too few data points
            lower_limit = clean_data.min()
            upper_limit = clean_data.max()
    
    else:  # parametric method
        # Assume normal distribution
        z_score = norm.ppf(1 - alpha/2)  # 1.96 for 95% CI
        lower_limit = mean_val - z_score * std_val
        upper_limit = mean_val + z_score * std_val
    
    # Comprehensive normality testing
    normality_results = perform_comprehensive_normality_tests(clean_data)
    
    # Calculate confidence intervals for reference limits (if sample size adequate)
    lower_ci = upper_ci = None
    if n >= 120:
        # Bootstrap-based confidence intervals for reference limits
        try:
            bootstrap_samples = 1000
            bootstrap_lowers = []
            bootstrap_uppers = []
            
            for _ in range(bootstrap_samples):
                bootstrap_data = np.random.choice(clean_data, size=n, replace=True)
                if method.lower() == "non-parametric":
                    boot_lower = np.percentile(bootstrap_data, lower_percentile)
                    boot_upper = np.percentile(bootstrap_data, upper_percentile)
                else:
                    boot_mean = np.mean(bootstrap_data)
                    boot_std = np.std(bootstrap_data, ddof=1)
                    boot_lower = boot_mean - z_score * boot_std
                    boot_upper = boot_mean + z_score * boot_std
                
                bootstrap_lowers.append(boot_lower)
                bootstrap_uppers.append(boot_upper)
            lower_ci = (np.percentile(bootstrap_lowers, 5), np.percentile(bootstrap_lowers, 95))
            upper_ci = (np.percentile(bootstrap_uppers, 5), np.percentile(bootstrap_uppers, 95))
            
        except Exception:
            pass
    
    return {
        'n': n,
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'cv_percent': cv_val,
        'lower_limit': lower_limit,
        'upper_limit': upper_limit,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'normality_results': normality_results,
        'method': method,
        'outliers_excluded': exclude_outliers,
        'outlier_method': outlier_method if exclude_outliers else None
    }

def create_enhanced_distribution_plot(analyte_data, ref_interval, selected_analyte):
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            f"Distribution - {selected_analyte}",
            "Normality Assessment",
            "Cumulative Distribution", 
            "Q-Q Plot",
            "Summary Statistics"
        ),
        specs=[
            [{"secondary_y": True}, {"type": "domain"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "table"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )   
    
    # Clean data
    clean_data = pd.to_numeric(analyte_data, errors='coerce').dropna()
    
    # 1. Main distribution histogram with KDE
    nbins = min(50, max(15, len(clean_data)//8))
    fig.add_trace(
        go.Histogram(
            x=clean_data,
            nbinsx=nbins,
            name="Population Data",
            marker_color="lightblue",
            opacity=0.7,
            histnorm="probability density",
            hovertemplate="Range: %{x}<br>Density: %{y}<extra></extra>",
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Add KDE curve
    try:
        
        kde = gaussian_kde(clean_data)
        x_range = np.linspace(clean_data.min(), clean_data.max(), 200)
        kde_values = kde(x_range)
        
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=kde_values,
                mode="lines",
                name="KDE",
                line=dict(color="blue", width=2),
                hovertemplate="Value: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>",
                showlegend=True
            ),
            row=1, col=1
        )
    except ImportError:
        pass  # Skip KDE if scipy not available
    
    # Get reference interval values
    lower_limit = ref_interval['lower_limit']
    upper_limit = ref_interval['upper_limit']
    
    # Add reference interval lines
    fig.add_vline(x=lower_limit, line=dict(color="red", width=2, dash="dash"), row=1, col=1)
    fig.add_vline(x=upper_limit, line=dict(color="red", width=2, dash="dash"), row=1, col=1)
    
    # 3. Normality assessment visualization
    normality_results = ref_interval.get('normality_results', {})
    if normality_results and len(normality_results) > 1:
        test_names = []
        p_values = []
        if 'shapiro_p' in normality_results:
            test_names.append('Shapiro-Wilk')
            p_values.append(normality_results['shapiro_p'])
        if 'jarque_bera_p' in normality_results:
            test_names.append('Jarque-Bera')
            p_values.append(normality_results['jarque_bera_p'])
        if test_names:
            colors = ['green' if p > 0.05 else 'red' for p in p_values]
            
            fig.add_trace(
                go.Bar(
                    x=test_names,
                    y=p_values,
                    marker_color=colors,
                    name="Normality Tests",
                    hovertemplate="Test: %{x}<br>p-value: %{y:.4f}<extra></extra>",
                    showlegend=False
                ),
                row=1, col=3
            )
            fig.add_hline(y=0.05, line=dict(color="black", width=1, dash="dash"), row=1, col=3)
    
    # 4. Cumulative distribution
    sorted_data = np.sort(clean_data)
    cumulative_prob = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
    
    fig.add_trace(
        go.Scatter(
            x=sorted_data,
            y=cumulative_prob,
            mode="lines",
            name="Empirical CDF",
            line=dict(color="blue", width=2),
            hovertemplate="Value: %{x:.3f}<br>Percentile: %{y:.1f}%<extra></extra>",
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add percentile markers
    percentiles = [2.5, 25, 50, 75, 97.5]
    percentile_values = np.percentile(clean_data, percentiles)
    
    fig.add_trace(
        go.Scatter(
            x=percentile_values,
            y=percentiles,
            mode="markers",
            name="Key Percentiles",
            marker=dict(color="red", size=8, symbol="diamond"),
            hovertemplate="Value: %{x:.3f}<br>Percentile: %{y}%<extra></extra>",
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 5. Q-Q Plot for normality assessment
    try:
        from scipy.stats import probplot
        (theoretical_quantiles, ordered_values), _ = probplot(clean_data, dist="norm", plot=None)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=ordered_values,
                mode="markers",
                name="Q-Q Plot",
                marker=dict(color="blue", size=4),
                hovertemplate="Theoretical: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>",
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Add reference line
        min_val = min(theoretical_quantiles.min(), ordered_values.min())
        max_val = max(theoretical_quantiles.max(), ordered_values.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Perfect Normal",
                showlegend=False
            ),
            row=2, col=2
        )
    except ImportError:
        pass
    
    # 6. Enhanced summary statistics table
    if n >= 120:
        quality = "Excellent"
    elif n >= 40:
        quality = "Good"
    elif n >= 20:
        quality = "Adequate"
    else:
        quality = "Insufficient"
    
    stats_data = [
        ["Statistic", "Value", "Interpretation"],
        ["Sample Size (N)", f"{ref_interval['n']}", quality],
        ["Lower Limit (2.5%)", f"{lower_limit:.4f}", "Values below potentially abnormal"],
        ["Upper Limit (97.5%)", f"{upper_limit:.4f}", "Values above potentially abnormal"],
        ["Reference Range", f"{lower_limit:.4f} - {upper_limit:.4f}", "95% of healthy population"],
        ["Mean", f"{ref_interval['mean']:.4f}", "Average value"],
        ["Median", f"{ref_interval['median']:.4f}", "Middle value"],
        ["Standard Deviation", f"{ref_interval['std']:.4f}", "Data spread"],
        ["CV%", f"{ref_interval['cv_percent']:.2f}%" if not np.isnan(ref_interval['cv_percent']) else "N/A", "Relative variability"]
    ]
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=stats_data[0],
                fill_color="orange",
                font=dict(color="white", size=14),
                align="left"
            ),
            cells=dict(
                values=list(zip(*stats_data[1:])),
                fill_color=[["#FFF5E1", "#FFF5E1", "#FFF5E1"]]*len(stats_data[1:]),
                align="left"
            )
        ),
        row=2, col=3
    )
    
    fig.update_layout(
        height=900,
        width=1100,
        title_text=f"Enhanced Distribution and Reference Interval Analysis - {selected_analyte}",
        title_x=0.5,
        showlegend=True,
        font=dict(family="Arial", size=12),
        margin=dict(t=100, b=50)
    )
    
    # Axis labels for plots
    fig.update_xaxes(title_text=selected_analyte, row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    
    fig.update_xaxes(title_text=selected_analyte, row=2, col=1)
    fig.update_yaxes(title_text="Percentile (%)", row=2, col=1)
    
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2)
    fig.update_yaxes(title_text="Observed Quantiles", row=2, col=2)
    
    # For normality bar plot
    fig.update_yaxes(title_text="p-value", row=1, col=3, range=[0, 1])
    
    return fig

def calculate_ages_from_dates(df, dob_col='Date of Birth', analysis_col='Date of Analysis',
                              add_all_units=True, handle_errors=True):
    df_copy = df.copy()
    summary = {
        'total_records': len(df_copy),
        'successful_conversions': 0,
        'date_errors': 0,
        'negative_ages': 0,
        'columns_added': []
    }
   
    try:
        # Convert date columns to datetime with error handling
        df_copy[dob_col] = pd.to_datetime(df_copy[dob_col], errors='coerce')
        df_copy[analysis_col] = pd.to_datetime(df_copy[analysis_col], errors='coerce')
       
        # Check for conversion errors
        dob_nulls = df_copy[dob_col].isnull().sum()
        analysis_nulls = df_copy[analysis_col].isnull().sum()
        summary['date_errors'] = max(dob_nulls, analysis_nulls)
       
        # Calculate age in days first
        age_days = (df_copy[analysis_col] - df_copy[dob_col]).dt.days
        
        valid_ages_mask = age_days >= 0
        negative_ages = (~valid_ages_mask).sum()
        summary['negative_ages'] = negative_ages
       
        if negative_ages > 0 and handle_errors:
            # Set negative ages to NaN
            age_days = age_days.where(valid_ages_mask)
       
        # Calculate age in years (most accurate)
        df_copy['Age (y)'] = age_days / 365.25
        summary['columns_added'].append('Age (y)')
       
        if add_all_units:
            # Calculate age in months (average days per month)
            df_copy['Age (m)'] = age_days / 30.44
            summary['columns_added'].append('Age (m)')
           
            # Calculate age in weeks
            df_copy['Age (w)'] = age_days / 7
            summary['columns_added'].append('Age (w)')
       
        # Count successful conversions
        summary['successful_conversions'] = df_copy['Age (y)'].notna().sum()
       
        # Round to reasonable precision
        for col in summary['columns_added']:
            df_copy[col] = df_copy[col].round(2)
       
        summary['success_rate'] = (summary['successful_conversions'] / summary['total_records']) * 100
        summary['status'] = 'success'
       
    except Exception as e:
        summary['status'] = 'error'
        summary['error_message'] = str(e)
        return df_copy, summary
   
    return df_copy, summary

def integrate_age_calculation(df, analyte_cols):
    if 'Age (y)' not in df.columns:
        st.info("🔢 Age (y) column not found. Calculating from Date of Birth and Date of Analysis...")
       
        df_with_ages, calc_summary = calculate_ages_from_dates(
            df,
            dob_col='Date of Birth',
            analysis_col='Date of Analysis',
            add_all_units=True,
            handle_errors=True
        )
        if calc_summary['status'] == 'success':
            st.success(f"✅ Age calculations completed! Added columns: {', '.join(calc_summary['columns_added'])}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", calc_summary['total_records'])
            with col2:
                st.metric("Successfully Calculated", calc_summary['successful_conversions'])
            with col3:
                st.metric("Success Rate", f"{calc_summary['success_rate']:.1f}%")
            with col4:
                st.metric("Date Errors", calc_summary['date_errors'])
            if calc_summary['date_errors'] > 0:
                st.warning(f"⚠️ {calc_summary['date_errors']} records had invalid dates")
            if calc_summary['negative_ages'] > 0:
                st.warning(f"⚠️ {calc_summary['negative_ages']} records had analysis date before birth date (set to NaN)")
           
            return df_with_ages
           
        else:
            st.error(f"❌ Error calculating age: {calc_summary.get('error_message', 'Unknown error')}")
            st.stop()
   
    return df

def calculate_age_partitioned_intervals_enhanced(df, analyte_cols, method="non-parametric", 
                                               min_samples_per_group=20, max_partitions=5):
    results = []

    if 'Age (y)' not in df.columns:
        st.warning("⚠️ Age (y) column not found. Cannot calculate age-partitioned intervals.")
        return pd.DataFrame()

    age_data = pd.to_numeric(df['Age (y)'], errors='coerce')
    valid_age_count = age_data.notna().sum()
    
    if valid_age_count == 0:
        st.warning("⚠️ No valid age data found. Cannot calculate age-partitioned intervals.")
        return pd.DataFrame()
    
    st.info(f"📊 Processing age-partitioned intervals for {len(analyte_cols)} analytes with {valid_age_count} valid age records")
    
    for analyte in analyte_cols:
        st.write(f"Processing {analyte}...")
        analyte_data = pd.to_numeric(df[analyte], errors='coerce')
        combined_data = pd.DataFrame({
            'analyte': analyte_data,
            'age': age_data
        })

        combined_data = combined_data.dropna()
        n_combined = len(combined_data)
        
        st.write(f"  - Combined data points: {n_combined}")
        
        if n_combined < min_samples_per_group * 2:
            st.warning(f"  - Insufficient data for {analyte} (n={n_combined}, need at least {min_samples_per_group * 2})")

            if n_combined >= 10:  
                ref_result = calculate_reference_interval_enhanced(combined_data['analyte'], method=method)
                if ref_result:
                    results.append({
                        'analyte': analyte,
                        'age_group': f"All ages (insufficient for partitioning)",
                        'age_range': f"{combined_data['age'].min():.1f}-{combined_data['age'].max():.1f}",
                        'n': ref_result['n'],
                        'lower_limit': ref_result['lower_limit'],
                        'upper_limit': ref_result['upper_limit'],
                        'mean': ref_result['mean'],
                        'median': ref_result['median'],
                        'std': ref_result['std'],
                        'cv_percent': ref_result['cv_percent'],
                        'quality_score': f'Single interval - only {n_combined} samples'
                    })
            continue
            
        combined_data = combined_data.sort_values('age').reset_index(drop=True)
        max_possible_partitions = min(n_combined // min_samples_per_group, max_partitions)
        
        if max_possible_partitions < 2:
            max_possible_partitions = 2  
        st.write(f"  - Max possible partitions: {max_possible_partitions}")
        best_partitions = min(max_possible_partitions, 3) 
        partition_boundaries = []
        for i in range(best_partitions + 1):
            quantile = i / best_partitions
            boundary = combined_data['age'].quantile(quantile)
            partition_boundaries.append(boundary)
        
        st.write(f"  - Age boundaries: {[f'{b:.1f}' for b in partition_boundaries]}")
        
        for i in range(best_partitions):
            if i == 0:
                partition_mask = combined_data['age'] <= partition_boundaries[i + 1]
            elif i == best_partitions - 1:
                partition_mask = combined_data['age'] > partition_boundaries[i]
            else:
                partition_mask = (combined_data['age'] > partition_boundaries[i]) & \
                               (combined_data['age'] <= partition_boundaries[i + 1])
            
            partition_data = combined_data[partition_mask]
            
            if len(partition_data) < min_samples_per_group:
                st.warning(f"  - Partition {i+1} has insufficient data ({len(partition_data)} samples)")
                continue
            
            # Calculate reference interval for this partition
            ref_result = calculate_reference_interval_enhanced(partition_data['analyte'], method=method)
            
            if ref_result:
                age_min = partition_data['age'].min()
                age_max = partition_data['age'].max()
                
                # Calculate quality score
                n_partition = len(partition_data)
                if n_partition >= 120:
                    quality_score = "Excellent"
                elif n_partition >= 40:
                    quality_score = "Good"
                elif n_partition >= 20:
                    quality_score = "Adequate"
                else:
                    quality_score = "Insufficient"
                
                results.append({
                    'analyte': analyte,
                    'age_group': f"Age Group {i+1}",
                    'age_range': f"{age_min:.1f}-{age_max:.1f} years",
                    'n': n_partition,
                    'lower_limit': ref_result['lower_limit'],
                    'upper_limit': ref_result['upper_limit'],
                    'mean': ref_result['mean'],
                    'median': ref_result['median'],
                    'std': ref_result['std'],
                    'cv_percent': ref_result['cv_percent'],
                    'quality_score': quality_score
                })
                
                st.write(f"  - ✅ Age Group {i+1}: {age_min:.1f}-{age_max:.1f} years, n={n_partition}")
            else:
                st.warning(f"  - ❌ Failed to calculate reference interval for Age Group {i+1}")
    
    results_df = pd.DataFrame(results)
    st.write(f"📋 Generated {len(results_df)} age-partitioned intervals")
    
    return results_df


# --- Main Application ---
uploaded_file = st.file_uploader(
    "📁 Choose a CSV file",
    type="csv",
    help="Upload a CSV file containing demographic data and analyte results from healthy individuals"
)

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
        
        # Display basic info about the uploaded file
        with st.expander("📋 Dataset Overview", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", df.shape[0])
            with col2:
                st.metric("Total Columns", df.shape[1])
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            st.write("**Column Information:**")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Null %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            st.dataframe(col_info, use_container_width=True)
            
            st.write("**First 5 rows:**")
            st.dataframe(df.head(), use_container_width=True)
        
        # Required columns check
        required_cols = ['Date of Analysis', 'Batch ID', 'Sample ID', 'Date of Birth']
        missing_required = [col for col in required_cols if col not in df.columns]
        
        if missing_required:
            st.error(f"❌ Missing required columns: {', '.join(missing_required)}")
            st.info("Please ensure your CSV contains all required columns: Date of Analysis, Batch ID, Sample ID, Date of Birth")
            st.stop()
        
        # Age calculation
        if 'Age (y)' not in df.columns:
            st.info("🔢 Age (y) column not found. Calculating from Date of Birth and Date of Analysis...")
            try:
                df['Date of Birth'] = pd.to_datetime(df['Date of Birth'], errors='coerce')
                df['Date of Analysis'] = pd.to_datetime(df['Date of Analysis'], errors='coerce')
                
                # Calculate age in years
                age_days = (df['Date of Analysis'] - df['Date of Birth']).dt.days
                df['Age (y)'] = age_days / 365.25
                
                # Also calculate age in months and weeks if not present
                if 'Age (m)' not in df.columns:
                    df['Age (m)'] = age_days / 30.44  # Average days per month
                if 'Age (w)' not in df.columns:
                    df['Age (w)'] = age_days / 7
                
                st.success("✅ Age calculations completed!")
                
            except Exception as e:
                st.error(f"❌ Error calculating age: {str(e)}")
                st.stop()
        
        # Identify analyte columns (numeric columns that aren't demographic)
        demographic_cols = ['Date of Analysis', 'Batch ID', 'Sample ID', 'Date of Birth', 
                          'Age (y)', 'Age (m)', 'Age (w)', 'Gender']
        potential_analyte_cols = [col for col in df.columns if col not in demographic_cols]
        
        # Convert analyte columns to numeric
        if potential_analyte_cols:
            st.info("🔄 Converting analyte columns to numeric values...")
            df_converted, conversion_summary = convert_to_numeric_enhanced(df, potential_analyte_cols)
            df = df_converted
            
            # Display conversion summary
            with st.expander("📊 Data Conversion Summary", expanded=False):
                for col, summary in conversion_summary.items():
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(f"**{col}**")
                    with col2:
                        st.metric("Original", summary['original_count'])
                    with col3:
                        st.metric("Converted", summary['numeric_count'])
                    with col4:
                        success_rate = summary['success_rate']
                        color = "normal" if success_rate >= 95 else "inverse"
                        st.metric("Success Rate", f"{success_rate:.1f}%", delta_color=color)
                    
                    if summary['failed_conversions'] > 0:
                        st.warning(f"⚠️ {summary['failed_conversions']} values could not be converted in {col}")
                        if summary['problematic_values']:
                            st.write(f"Examples of problematic values: {', '.join(summary['problematic_values'][:3])}")
            
            # Filter to only successfully converted numeric columns
            analyte_cols = [col for col in potential_analyte_cols 
                          if col in df.columns and df[col].dtype in ['float64', 'int64']]
            
            if not analyte_cols:
                st.error("❌ No numeric analyte columns found after conversion!")
                st.stop()
                
            st.success(f"✅ Found {len(analyte_cols)} analyte columns: {', '.join(analyte_cols)}")
        
        else:
            st.error("❌ No potential analyte columns identified!")
            st.stop()
        
        # --- Sidebar Controls ---
        st.sidebar.header("🎛️ Analysis Options")
        
        # Gender filter
        if 'Gender' in df.columns:
            gender_options = ['All'] + sorted(df['Gender'].dropna().unique().tolist())
            selected_gender = st.sidebar.selectbox("👤 Filter by Gender", gender_options)
            
            if selected_gender != 'All':
                df = df[df['Gender'] == selected_gender]
                st.sidebar.info(f"Filtered to {selected_gender}: {len(df)} records")
        
        # Age filter
        if 'Age (y)' in df.columns:
            age_data = pd.to_numeric(df['Age (y)'], errors='coerce').dropna()
            if len(age_data) > 0:
                min_age, max_age = float(age_data.min()), float(age_data.max())
                age_range = st.sidebar.slider(
                    "🎂 Age Range (years)",
                    min_value=min_age,
                    max_value=max_age,
                    value=(min_age, max_age),
                    step=0.1
                )
                
                # Apply age filter
                age_mask = (pd.to_numeric(df['Age (y)'], errors='coerce') >= age_range[0]) & \
                          (pd.to_numeric(df['Age (y)'], errors='coerce') <= age_range[1])
                df = df[age_mask]
                st.sidebar.info(f"Age filtered: {age_mask.sum()} records")
        
        # Batch filter
        if 'Batch ID' in df.columns:
            batch_options = ['All'] + sorted(df['Batch ID'].dropna().astype(str).unique().tolist())
            selected_batches = st.sidebar.multiselect("🧪 Filter by Batch ID", batch_options, default=['All'])
            
            if 'All' not in selected_batches and selected_batches:
                df = df[df['Batch ID'].astype(str).isin(selected_batches)]
                st.sidebar.info(f"Batch filtered: {len(df)} records")
        
        # Analysis method
        method = st.sidebar.selectbox(
            "📊 Reference Interval Method",
            ["Non-parametric", "Parametric"],
            help="Non-parametric (percentile-based) is recommended for most cases"
        )
        
        # Outlier handling
        st.sidebar.subheader("🎯 Outlier Detection")
        exclude_outliers = st.sidebar.checkbox("Exclude Outliers", value=False)
        
        if exclude_outliers:
            outlier_method = st.sidebar.selectbox(
                "Outlier Detection Method",
                ["3sd", "iqr"],
                help="3sd: 3 standard deviations from mean, IQR: 1.5 × interquartile range"
            )
        else:
            outlier_method = "3sd"
        
        # Advanced options
        with st.sidebar.expander("🔧 Advanced Options"):
            confidence_level = st.slider("Confidence Level", 0.90, 0.99, 0.95, 0.01)
            show_age_partitioned = st.checkbox("Calculate Age-Partitioned Intervals", value=False)
            
            if show_age_partitioned:
                min_samples_per_group = st.number_input("Min Samples per Age Group", 15, 50, 20)
                max_partitions = st.number_input("Max Age Partitions", 2, 10, 5)
        
        # --- Main Analysis ---
        st.header("🔬 Reference Interval Results")
        
        # Calculate reference intervals for all analytes
        results_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, analyte in enumerate(analyte_cols):
            status_text.text(f"Processing {analyte}... ({i+1}/{len(analyte_cols)})")
            
            # Get clean data for this analyte
            analyte_data = pd.to_numeric(df[analyte], errors='coerce').dropna()
            
            if len(analyte_data) < 3:
                st.warning(f"⚠️ Insufficient data for {analyte} (n={len(analyte_data)})")
                continue
            
            # Calculate reference interval
            ref_result = calculate_reference_interval_enhanced(
                analyte_data,
                method=method.lower(),
                confidence_level=confidence_level,
                exclude_outliers=exclude_outliers,
                outlier_method=outlier_method
            )
            
            if ref_result:
                # Determine quality assessment
                n = ref_result['n']
                if n >= 120:
                    quality = "✅ Excellent"
                    quality_color = "normal"
                elif n >= 40:
                    quality = "🟡 Good"
                    quality_color = "normal"
                elif n >= 20:
                    quality = "🟠 Adequate"
                    quality_color = "off"
                else:
                    quality = "🔴 Insufficient"
                    quality_color = "inverse"
                
                # Normality assessment
                normality = ref_result.get('normality_results', {})
                is_normal = normality.get('overall_normal', 'Unknown')
                normality_text = "✅ Normal" if is_normal == True else ("❌ Non-normal" if is_normal == False else "❓ Unknown")
                
                results_data.append({
                    'Analyte': analyte,
                    'N': n,
                    'Lower Limit': ref_result['lower_limit'],
                    'Upper Limit': ref_result['upper_limit'],
                    'Reference Interval': f"{ref_result['lower_limit']:.4f} - {ref_result['upper_limit']:.4f}",
                    'Mean': ref_result['mean'],
                    'Median': ref_result['median'],
                    'SD': ref_result['std'],
                    'CV%': ref_result['cv_percent'],
                    'Quality': quality,
                    'Normal Distribution': normality_text,
                    'Method': ref_result['method'].title()
                })
            
            progress_bar.progress((i + 1) / len(analyte_cols))
        
        status_text.text("Analysis complete!")
        progress_bar.empty()
        status_text.empty()
        
        if results_data:
            # Display results table
            results_df = pd.DataFrame(results_data)
            
            st.subheader("📋 Summary Table")
            
            # Create interactive table with color coding
            st.dataframe(
                results_df,
                use_container_width=True,
                column_config={
                    "N": st.column_config.NumberColumn("Sample Size", format="%d"),
                    "Lower Limit": st.column_config.NumberColumn("Lower Limit", format="%.4f"),
                    "Upper Limit": st.column_config.NumberColumn("Upper Limit", format="%.4f"),
                    "Mean": st.column_config.NumberColumn("Mean", format="%.4f"),
                    "Median": st.column_config.NumberColumn("Median", format="%.4f"),
                    "SD": st.column_config.NumberColumn("Standard Deviation", format="%.4f"),
                    "CV%": st.column_config.NumberColumn("CV%", format="%.2f"),
                    "Quality": st.column_config.TextColumn("Quality Assessment"),
                    "Normal Distribution": st.column_config.TextColumn("Normality"),
                }
            )
            
            # Download button for results
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name=f"reference_intervals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Individual analyte analysis
            st.subheader("🔍 Detailed Analyte Analysis")
            
            selected_analyte = st.selectbox(
                "Select analyte for detailed analysis:",
                analyte_cols,
                help="Choose an analyte to view detailed distribution plots and statistics"
            )
            
            if selected_analyte:
                # Get the reference interval for selected analyte
                analyte_data = pd.to_numeric(df[selected_analyte], errors='coerce').dropna()
                ref_result = calculate_reference_interval_enhanced(
                    analyte_data,
                    method=method.lower(),
                    confidence_level=confidence_level,
                    exclude_outliers=exclude_outliers,
                    outlier_method=outlier_method
                )
                
                if ref_result:
                    # Create comprehensive distribution plot
                    fig = create_enhanced_distribution_plot(analyte_data, ref_result, selected_analyte)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional detailed statistics
                    with st.expander("📊 Detailed Statistics", expanded=False):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Sample Size", ref_result['n'])
                            st.metric("Mean", f"{ref_result['mean']:.4f}")
                        
                        with col2:
                            st.metric("Median", f"{ref_result['median']:.4f}")
                            st.metric("Standard Deviation", f"{ref_result['std']:.4f}")
                        
                        with col3:
                            st.metric("CV%", f"{ref_result['cv_percent']:.2f}%" if not np.isnan(ref_result['cv_percent']) else "N/A")
                            st.metric("Lower Limit (2.5%)", f"{ref_result['lower_limit']:.4f}")
                        
                        with col4:
                            st.metric("Upper Limit (97.5%)", f"{ref_result['upper_limit']:.4f}")
                            st.metric("Range Width", f"{ref_result['upper_limit'] - ref_result['lower_limit']:.4f}")
                        
                        # Normality test results
                        normality_results = ref_result.get('normality_results', {})
                        if normality_results and len(normality_results) > 1:
                            st.subheader("🔬 Normality Test Results")
                            
                            normality_data = []
                            if 'shapiro_p' in normality_results:
                                shapiro_result = "Pass" if normality_results['shapiro_normal'] else "Fail"
                                normality_data.append(["Shapiro-Wilk", f"{normality_results['shapiro_p']:.6f}", shapiro_result])
                            
                            if 'jarque_bera_p' in normality_results:
                                jb_result = "Pass" if normality_results['jarque_bera_normal'] else "Fail"
                                normality_data.append(["Jarque-Bera", f"{normality_results['jarque_bera_p']:.6f}", jb_result])
                            
                            if 'anderson_darling_normal' in normality_results:
                                ad_result = "Pass" if normality_results['anderson_darling_normal'] else "Fail"
                                normality_data.append(["Anderson-Darling", f"{normality_results['anderson_darling_stat']:.4f}", ad_result])
                            
                            if normality_data:
                                normality_df = pd.DataFrame(normality_data, columns=["Test", "Statistic/p-value", "Result (α=0.05)"])
                                st.dataframe(normality_df, use_container_width=True)
                                
                                st.info("📝 **Interpretation**: 'Pass' means the data is consistent with normal distribution (p > 0.05 or statistic < critical value)")
            
            # Age-partitioned intervals
            if show_age_partitioned and 'Age (y)' in df.columns:
                st.subheader("👶👴 Age-Partitioned Reference Intervals")
                
                with st.spinner("Calculating age-partitioned intervals..."):
                    age_partitioned_results = calculate_age_partitioned_intervals_enhanced(
                        df, analyte_cols, method=method.lower(),
                        min_samples_per_group=min_samples_per_group,
                        max_partitions=max_partitions
                    )
                
                if not age_partitioned_results.empty:
                    st.dataframe(age_partitioned_results, use_container_width=True)
                    age_csv_data = age_partitioned_results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Age-Partitioned Results",
                        data=age_csv_data,
                        file_name=f"age_partitioned_intervals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ Unable to calculate age-partitioned intervals. Insufficient data for age grouping.")
        
        else:
            st.error("❌ No results could be calculated. Please check your data and try again.")
    
    except Exception as e:
        st.error(f"❌ An error occurred while processing the file: {str(e)}")
        st.info("Please check that your CSV file is properly formatted and contains the required columns.")

else:
    st.info("👆 Please upload a CSV file to begin the analysis")
    
    # Show example data format
    with st.expander("📋 Example Data Format", expanded=False):
        example_data = pd.DataFrame({
            'Date of Analysis': ['2024-01-15', '2024-01-15', '2024-01-16'],
            'Batch ID': ['B001', 'B001', 'B002'],
            'Sample ID': ['S001', 'S002', 'S003'],
            'Date of Birth': ['1990-05-20', '1985-03-10', '1992-11-30'],
            'Gender': ['M', 'F', 'M'],
            'Glucose': [5.2, 4.8, 5.5],
            'Cholesterol': [4.2, 3.9, 4.6],
            'Hemoglobin': [14.5, 12.8, 15.1]
        })
        
        st.write("**Example CSV structure:**")
        st.dataframe(example_data, use_container_width=True)
        
        # Create downloadable example
        example_csv = example_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Example CSV",
            data=example_csv,
            file_name="example_reference_interval_data.csv",
            mime="text/csv"
        )
