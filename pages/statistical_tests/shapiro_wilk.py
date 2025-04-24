import streamlit as st
import pandas as pd
import scipy.stats as stats

def run():
    st.title("\U0001F4CA Shapiro-Wilk Normality Test")

    st.markdown("""
    The **Shapiro-Wilk test** is used to test the normality of a dataset. It tests the null hypothesis that the data is normally distributed. If the p-value is small (typically < 0.05), the null hypothesis is rejected, and we conclude that the data is not normally distributed.

    ### \U0001F4C2 Data Upload Requirements:
    - Upload a **CSV file** containing your experimental data for the test.
    - Ensure that the column you wish to test for normality contains continuous numerical values.
    """)

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Data Preview:")
        st.write(df.head())

        column_name = st.selectbox("Select the column to test for normality", df.columns)

        if pd.api.types.is_numeric_dtype(df[column_name]):
            stat, p_value = stats.shapiro(df[column_name].dropna())

            st.subheader(f"Shapiro-Wilk Test Results for {column_name}")
            st.write(f"Test Statistic: {stat:.5f}")
            st.write(f"P-Value: {p_value:.5f}")

            if p_value > 0.05:
                st.success("The data appears to be normally distributed (fail to reject H0).")
            else:
                st.error("The data does not appear to be normally distributed (reject H0).")
        else:
            st.error("Please select a numeric column for normality testing.")
    else:
        st.info("Please upload a CSV file to get started.")
