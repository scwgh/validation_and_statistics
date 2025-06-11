import streamlit as st
import pandas as pd
import plotly.express as px

st.title("EQA Comparison")

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
        analytes = df.columns[7:]
        selected_analyte = st.selectbox("Select Analyte", analytes)
        analyzers = df["Analyser"].unique()
        if len(analyzers) < 2:
            st.warning("Need at least two analyzers in the dataset.")
            return

        analyzer_1 = st.selectbox("Select Reference Analyzer (Analyser 1)", analyzers, key="ref")
        remaining_analyzers = [a for a in analyzers if a != analyzer_1]
        analyzer_2 = st.selectbox("Select New Analyzer (Analyser 2)", remaining_analyzers, key="new")

        units = st.selectbox(
            "Select Units for Analytes",
            options=units_list, 
            index=0
        )
    st.subheader("Comparison Plots and Stats")

    for analyte in analyte_cols:
        expected_col = f"Expected {analyte}"
        comparison_df = df[[analyte, expected_col, 'Sample ID']].dropna()

        comparison_df["% Difference"] = (
            (comparison_df[analyte] - comparison_df[expected_col]) / comparison_df[expected_col]
        ) * 100

        # Plot
        fig = px.scatter(
            comparison_df,
            x=expected_col,
            y=analyte,
            hover_data=["Sample ID", "% Difference"],
            title=f"{analyte} - Measured vs Expected",
            labels={expected_col: "Expected", analyte: "Measured"},
            color=comparison_df["% Difference"].abs() > tolerance
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table with flagged values
        flagged = comparison_df[comparison_df["% Difference"].abs() > tolerance]
        if not flagged.empty:
            st.warning(f"{len(flagged)} samples exceed ±{tolerance}% for {analyte}")
            st.dataframe(flagged)
