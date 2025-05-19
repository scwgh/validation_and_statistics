import streamlit as st
import importlib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import apply_app_styling

# Page setup
st.set_page_config(
    page_title="Linearity",
    page_icon="📏",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_app_styling()

st.title("📏 Linearity")

# with st.spinner("Loading the analysis tools... Please wait!"):
#     st.success("Analysis tools loaded successfully! 🎉 " \
#     "\n Let's get started with your validation analysis!")

with st.expander("📘 Why assess linearity?", expanded=True):
    st.markdown("""
    **Linearity** refers to the ability of an analytical method to produce results that are directly proportional to the concentration of the analyte in the sample. 
                \n The linear range experiment requires enoigh of each specimen to prepare dilutions and to carry out testing with 5 or more concentrations. 
                \n A test of linearity starts with a plot of the measured values against the corresponding reference standards to see whether the points fall along a straight line. 
                \n **Calibration Curve**:
                    \n > In the context of a calibration curve, we use linearity studies to predict future measurements made with the same instrument. Applying this to calibration results, a calibration line is created using the inverse of a linear model:""")
    st.latex(r''' {Y} = \alpha + \beta{x} + \epsilon ''')
    st.markdown("""> This may be re-worked to provide the calibrated value:""")
    st.latex(r'''\bar{X} = \frac{Y - \hat{\alpha}}{\hat{\beta}}''')

    st.markdown("""At least one measurement is needed for each standard. The linearity assumption in linear regression means that teh relationship between the independent and dependent variable is a straight line. If this is met, the instrument performance is linear. Statistical control in this context implies not only that the measurements are repeatable within certain limits, but that the instrument response also remains linear. 
                \n It is a critical aspect of method validation, ensuring that the method provides accurate and reliable results across the entire range of concentrations.""")
with st.expander("📘 Investigating issues with linearity", expanded=False):   
    st.markdown("""
    Calibration does not always eliminate bias. There are several important factors to consider when investigating poor linearity:

    - **Poor Precision**: Instrument imprecision or day-to-day variability can lead to unreliable calibration. Precision should be assessed before selecting an instrument.
    - **Outliers**: Extreme data points, especially at calibration range endpoints, can distort the curve. Isolated outliers should be removed, and inconsistent daily results must be reviewed.
    - **Operator Bias**: Different operators may introduce systematic biases. If significant, consider retraining or creating separate calibration curves per operator.
    - **System Instability**: Instrument drift over time can invalidate calibration. Regular statistical monitoring is essential.
    - **Unseen Day-to-Day Variation**: Aggregated plots may hide daily inconsistencies. Fine-grained plots can help reveal such hidden variability.""")

with st.expander("📘 Instructions:"):
    st.markdown("""
    1. Upload a CSV file containing your standard curve data.
    2. Ensure the file includes columns for `Concentration`, `Response`, and optionally `Identifier`.
    3. Select the appropriate columns for analysis.
    4. Click the button below to run the standard curve analysis.
    """)

results_df = None

with st.expander("📄 Upload Your CSV File:", expanded=True):
    st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
    uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"], label_visibility="collapsed")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    else:
        df = None
        st.info("Awaiting file upload...")

if df is not None:
    st.subheader("📖 Data Preview:")
    st.dataframe(df)

    units = st.selectbox(
        "Select Units for Analytes",
        options=["μmol/L", "mmol/L", "mg/dL", "g/L", "ng/mL"], 
        index=0
    )

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_columns) < 2:
        st.error("❌ Not enough numeric columns found for X and Y selection.")
    else:
        x_axis = st.selectbox("Select the X-axis (e.g., Standard Concentration)", numeric_columns)

        # Suggest the next column as the default for Y-axis
        x_index = numeric_columns.index(x_axis)
        default_y_index = x_index + 1 
        y_axis = st.selectbox("Select the Y-axis (e.g., Measured Value)", numeric_columns, index=default_y_index)

        identifier_column = st.selectbox("Optional: Select an identifier column (e.g., Sample ID)", [None] + df.columns.tolist())

        clean_df = df[[x_axis, y_axis] + ([identifier_column] if identifier_column else [])].dropna()

        if clean_df.empty:
            st.error("❌ The selected columns contain no valid numeric data.")
        else:
            try:
                x = clean_df[x_axis].to_numpy()
                y = clean_df[y_axis].to_numpy()

                slope, intercept = np.polyfit(x, y, 1)
                fitted_values = slope * x + intercept
                residuals = y - fitted_values
                r_squared = 1 - (np.sum(residuals**2) / np.sum((y - np.mean(y))**2))

                hover_text = (
                    clean_df[identifier_column].astype(str) + "<br>"
                    + x_axis + ": " + clean_df[x_axis].astype(str) + "<br>"
                    + y_axis + ": " + clean_df[y_axis].astype(str)
                ) if identifier_column else None

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='markers',
                    name='Data Points',
                    marker=dict(color='blue'),
                    text=hover_text,
                    hoverinfo='text' if identifier_column else 'x+y'
                ))
                fig.add_trace(go.Scatter(
                    x=x,
                    y=fitted_values,
                    mode='lines',
                    name=f"Fit: y = {slope:.2f}x + {intercept:.2f}<br>R² = {r_squared:.4f}",
                    line=dict(color='red')
                ))
                fig.update_layout(
                    title="Standard Curve (Linear Fit)",
                    xaxis_title=f"{x_axis} ({units})",
                    yaxis_title=f"{y_axis} ({units})",
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig)

                if r_squared >= 0.99:
                    interpretation = "Excellent linearity — the response is highly consistent with the standard concentrations."
                elif r_squared >= 0.95:
                    interpretation = "Good linearity — results are acceptable, but further verification may be considered."
                elif r_squared >= 0.90:
                    interpretation = "Moderate linearity — further investigation may be needed for accuracy at extreme points."
                else:
                    interpretation = "Poor linearity — data may not be reliable for quantitative analysis."

                st.markdown(f"🧠 **Interpretation:** \n{interpretation}")

                results_df = clean_df.copy()
                results_df["Fitted Value"] = fitted_values
                results_df["Residuals"] = residuals

            except np.linalg.LinAlgError:
                st.error("❌ Linear fitting failed due to numerical instability in the data.")
            except TypeError:
                st.error("❌ Ensure X and Y axis selections are numeric and 1D.")

            with st.expander("📊 Recovery"):
                st.subheader("📊 Recovery Calculation")
                selectable_columns = df.columns[6:]
                expected_column = st.selectbox("Select the Expected (e.g., C26)", selectable_columns)
                calculated_column = st.selectbox("Select the Calculated (e.g., Calculated C26)", selectable_columns)

                clean_df["Sample ID"] = df["Sample ID"] if "Sample ID" in df.columns else "N/A"
                clean_df["Expected (" + str(units) + ")"] = df[expected_column]
                clean_df["Calculated (" + str(units) + ")"] = df[calculated_column]

                clean_df["Recovery (%)"] = np.where(clean_df["Expected (" + str(units) + ")"] > 0, (clean_df["Calculated (" + str(units) + ")"] / clean_df["Expected (" + str(units) + ")"]) * 100, np.nan)

                show_mean = st.checkbox("Show Recovery Summary (per Sample)")

                if show_mean:
                    mean_df = clean_df.groupby("Sample ID").agg({"Expected (" + str(units) + ")": "mean", "Calculated (" + str(units) + ")": "mean", "Recovery (%)": "mean"}).reset_index().round(2)
                    st.markdown("### 📊 Mean Recovery Table")
                    st.dataframe(mean_df)
                else:
                    st.markdown("### 📊 Recovery Table")
                    st.dataframe(clean_df[["Sample ID", x_axis, y_axis, "Expected (" + str(units) + ")", "Calculated (" + str(units) + ")", "Recovery (%)"]])

            if results_df is not None:
                st.markdown("### 📅 Download Results")
                st.markdown("Download the standard curve results including fitted values and residuals.")

                st.download_button(
                    label="⬇ Download Results",
                    data=results_df.to_csv(index=False).encode('utf-8'),
                    file_name="standard_curve_results.csv",
                    mime="text/csv"
                )



