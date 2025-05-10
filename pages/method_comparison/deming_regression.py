import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.odr import ODR, RealData, Model
import plotly.graph_objects as go
from utils import apply_app_styling

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
        1. Upload a CSV with `Material`, `Analyser`, `Sample ID`, and analyte columns.
        2. Choose two analyzers for comparison.
        3. Choose a material type and a unit.
        4. Run Deming Regression to view plots and download results.
        """)

    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
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
                    options=["μmol/L", "mmol/L", "mg/dL", "g/L", "ng/mL"],
                    index=0
                )

                # Step 5: Run Deming Regression for selected analytes
                all_results = []
                for selected_analyte in selected_analytes:
                    result = deming_regression_analysis(df, analyzer_1, analyzer_2, selected_material, units, selected_analyte)
                    if result:
                        all_results.extend(result)

                # Step 6: Display results in tabbed summary table
                if all_results:
                    results_df = pd.DataFrame(all_results)
                    material_tabs = st.radio("Select Material Type", ["EQA", "Patient"], index=0)
                    filtered_results = results_df[results_df['Material'] == material_tabs]

                    st.dataframe(filtered_results)

                    # Download button
                    st.download_button(
                        label="⬇ Download Results (CSV)",
                        data=filtered_results.to_csv(index=False).encode('utf-8'),
                        file_name=f"deming_regression_results_{material_tabs}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠ No valid data for regression.")

def deming_regression_analysis(df, analyzer_1, analyzer_2, selected_material, units, selected_analyte):
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

    def linear(B, x): return B[0] * x + B[1]
    model = Model(linear)
    odr_data = RealData(x, y)
    odr = ODR(odr_data, model, beta0=[1, 0])
    output = odr.run()

    slope, intercept = output.beta
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # Calculate standard errors for slope and intercept
    dof = len(x) - 2  # degrees of freedom
    mse = ss_res / dof  # mean square error
    var_x = np.var(x, ddof=1)  # variance of x

    se_slope = np.sqrt(mse / np.sum((x - np.mean(x)) ** 2))
    se_intercept = np.sqrt(mse * (1/len(x) + np.mean(x)**2 / np.sum((x - np.mean(x)) ** 2)))

    # t-value for 95% confidence interval (two-tailed)
    t_val = stats.t.ppf(0.975, dof)

    # Confidence intervals for slope and intercept
    ci_slope = t_val * se_slope
    ci_intercept = t_val * se_intercept

    # Confidence bounds for the regression line
    x_line = np.linspace(min(x), max(x), 100)
    y_line = slope * x_line + intercept

    # Upper and lower bounds of the confidence intervals
    y_upper = y_line + ci_slope * x_line + ci_intercept
    y_lower = y_line - ci_slope * x_line - ci_intercept

    results_list.append({
        'Analyte': selected_analyte,
        'Material': selected_material,
        'Analyzer 1': analyzer_1,
        'Analyzer 2': analyzer_2,
        'Slope': round(slope, 4),
        'Intercept': round(intercept, 4),
        'R²': round(r_squared, 4),
        'n': len(pivot),
        'Critical t-value': round(t_val, 4),
        'SE (Slope)': round(se_slope, 4)
    })

    # Plot the results (optional)
    line_name = f"y = {slope:.2f}x + {intercept:.2f} (R² = {r_squared:.4f})"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Samples',
        marker=dict(color='steelblue'),
        hovertemplate=f"{analyzer_1}: %{{x:.2f}} {units}<br>{analyzer_2}: %{{y:.2f}} {units}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        name=line_name,  # Use line_name with R² in the legend
        line=dict(color='red')
    ))

    # Add confidence intervals (shaded region)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_line, x_line[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 99, 71, 0.3)',  # Light red for confidence interval
        line=dict(color='rgba(255, 99, 71, 0)'),
        name='95% Confidence Interval',
        showlegend=True
    ))

    fig.update_layout(
        title=f"Deming Regression for {selected_analyte} ({selected_material})",
        xaxis_title=f"{analyzer_1} ({units})",
        yaxis_title=f"{analyzer_2} ({units})",
        height=500,
        plot_bgcolor='white'
    )
    st.plotly_chart(fig, use_container_width=True)

    return results_list
