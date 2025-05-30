import streamlit as st
import pandas as pd
import plotly.express as px
from utils import apply_app_styling

# --- Page Configuration ---
st.set_page_config(
    page_title="Limit of Quantitation (LOQ)",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_app_styling()

st.title("🚀 Limit of Quantitation (LOQ)")

# --- Method Explanation ---
with st.expander("📘 What is Limit of Quantitation Analysis?", expanded=True):
    st.markdown("""
    **Limit of Quantitation (LOQ)** is the lowest amount of analyte in a sample that can be quantitatively determined with suitable precision and accuracy.
    
    It's typically calculated from blank sample variability using the formula:
    
    **LOQ = 10 × standard deviation (SD) of blanks**

    This method helps ensure the reliability of analytical results, particularly in trace-level detection.
    """)

# --- Instructions ---
with st.expander("📘 Instructions"):
    st.markdown("""
    1. Upload a CSV file with your analyte data.
    2. Ensure the file includes repeated blank samples labeled in the `Material` column.
    3. Select the analyte columns (numeric values expected).
    4. View the calculated LOQ results and visualizations.
    """)

# --- File Upload Function ---
def upload_data():
    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"], label_visibility="collapsed")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.markdown("### 📋 Data Preview")
            st.dataframe(df.head())
            return df
        else:
            st.info("Awaiting file upload...")
            return None

# --- Upload Data ---
df = upload_data()

# --- Process and Display LOQ ---
if df is not None:
    # Filter for blank data
    blank_data = df[df['Material'] == 'Blank']

    # Get only analyte columns (e.g., 4, 7, 10, 13...) based on structure
    analyte_columns = df.columns[4::3]

    # Initialize a dictionary to store results
    loq_results = {'Analyte': [], 'Mean': [], 'SD': [], '10*SD (LOQ)': []}

    for analyte in analyte_columns:
        valid_data = pd.to_numeric(blank_data[analyte], errors='coerce').dropna()
        if valid_data.empty:
            st.info(f"No valid data for {analyte}, skipping...")
            continue

        mean = round(valid_data.mean(), 5)
        sd = round(valid_data.std(), 5)
        loq = round(10 * sd, 5)

        loq_results['Analyte'].append(analyte)
        loq_results['Mean'].append(mean)
        loq_results['SD'].append(sd)
        loq_results['10*SD (LOQ)'].append(loq)

    loq_df = pd.DataFrame(loq_results)

    st.subheader("📊 LOQ Results")
    st.dataframe(loq_df)

    # Download button
    loq_csv = loq_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download LOQ Results CSV", loq_csv, "loq_results.csv", "text/csv")

    # Plotting
    st.subheader("📈 LOQ Scatter Plot")
    fig = px.scatter(
        loq_df,
        x="Analyte",
        y="10*SD (LOQ)",
        text="10*SD (LOQ)",
        color_discrete_sequence=["seagreen"],
        size_max=60
    )

    fig.update_traces(textposition='top center', marker=dict(size=12))
    fig.update_layout(
        title="Limit of Quantitation (LOQ) for Analytes",
        xaxis_title="Analyte",
        yaxis_title="LOQ (10 × SD)",
        plot_bgcolor="white",
        font=dict(size=14),
        title_font=dict(size=18),
        xaxis_tickangle=45,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='lightgray')

    st.plotly_chart(fig, use_container_width=True)
