import streamlit as st
import pandas as pd
from utils import apply_app_styling

# Page setup
st.set_page_config(
    page_title="Limits",
    page_icon="♾️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_app_styling()
st.title("♾️ Limits")

# --- Method Explanation ---
with st.expander("📘 What are limits?", expanded=True):
    st.markdown("""
    It is important to characterise the analytical performance of a laboratory assay to determine if it is "_fit for purpose_". This includes understanding the limits of detection (LOD), quantification (LOQ), and blank (LOB) for the analytes of interest. 
    \n LOB and LOD are important to discriminate between the presence or absence of an analyte; whereas LOQ is important for clinical diagnosis and management. 
    \n **LOD**, or **Limit of Detection**, refers to the **lowest concentration of an analyte** that can be reliably distinguished from background noise or a blank signal—but not necessarily quantified with accuracy or precision.
    """)
    st.latex(r"\text{LOD} = \text{LOB} + 1.645 \cdot \sigma_{\text{low concentration sample}}")

    st.markdown("""**LOB**, or **Limit of Blank**, is the highest concentration of analyte that is likely to be found in a blank sample.
    """)
    st.latex(r"\text{LOB} = \mu_{\text{blank}} + 1.645 \cdot \sigma_{\text{blank}}")

    st.markdown("""**LOQ**, or **Limit of Quantification**, is the lowest concentration at which the analyte can be quantitatively detected with acceptable precision and accuracy.    
    """)
    st.latex(r"\text{LOQ} = 10 \cdot \sigma_{\text{low concentration sample}}")

# --- Instructions ---
with st.expander("📘 Instructions"):
    st.markdown("""***For LOB analysis:***""")
    st.markdown("""
    1. Upload a CSV file with your limit data.
    2. Ensure the file includes repeated blank samples labeled in the `Material` column.
    3. Select the analyte columns (numeric values expected).
    4. The calculation results will appear below once performed.
    """)
    st.markdown("""***For LOD/LOQ analysis:***""")
    st.markdown("""
    1. Upload a CSV file with your limit data.
    2. Ensure the file includes repeated low concentration samples labeled in the `Material` column (e.g., LowConc1).
    3. Select the analyte columns (numeric values expected).
    4. The calculation results will appear below once performed.
    """)

# --- File Upload Function ---
def upload_data():
    with st.expander("📤 Upload Your CSV File", expanded=True):
        st.markdown("Upload a CSV containing your analyte data. Ensure it includes the following columns: `Material`, `Analyser`, and `Sample ID`.")
        uploaded_file = st.file_uploader("Choose a file to get started", type=["csv"], label_visibility="collapsed")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.markdown("### 📖 Data Preview")
            st.dataframe(df.head())
            return df
        else:
            st.info("Awaiting file upload...")
            return None

# --- Calculation Logic for LOB ---
def calculate_lob(df):
    # --- Identify analyte columns ---
    ignore_cols = ['Date', 'Analyser', 'Material', 'Sample ID']
    analyte_columns =  [df.columns[i] for i in range(4, len(df.columns), 3)] 

    # --- Filter blank data ---
    blank_data = df[df['Material'].str.lower() == 'blank']

    results = {
        'Analyte': [],
        'Blank Mean': [],
        'Blank SD': [],
        'LOB': []
    }

    for analyte in analyte_columns:
        blank_vals = pd.to_numeric(blank_data[analyte], errors='coerce').dropna()

        if blank_vals.empty:
            continue

        blank_mean = round(blank_vals.mean(), 5)
        blank_sd = round(blank_vals.std(), 5)

        lob = round(blank_mean + 1.645 * blank_sd, 5)

        results['Analyte'].append(analyte)
        results['Blank Mean'].append(blank_mean)
        results['Blank SD'].append(blank_sd)
        results['LOB'].append(lob)

    result_df = pd.DataFrame(results)
    st.subheader("📊 Limit of Blank (LOB) Summary")
    st.dataframe(result_df)

    # st.download_button(
    #     label="⬇️ Download LOB Results as CSV",
    #     data=result_df.to_csv(index=False),
    #     file_name="lob_results.csv",
    #     mime="text/csv"
    # )

# --- Calculation Logic for LOD & LOQ ---
def calculate_lod_loq(df):
    # --- Identify analyte columns ---
    ignore_cols = ['Date', 'Analyser', 'Material', 'Sample ID']
    analyte_columns = [df.columns[i] for i in range(4, len(df.columns), 3)]

    # --- Filter low concentration data ---
    low_data = df[df['Sample ID'].str.lower().str.contains('low')]

    results = {
        'Analyte': [],
        'Low SD': [],
        'Slope': [],
        'LOD': [],
        'LOQ': []
    }

    for analyte in analyte_columns:
        # Attempt to find associated standard concentration column
        conc_col = f'{analyte}'
        if conc_col not in df.columns:
            st.warning(f"⚠️ No matching concentration column found for {analyte}. Expected column: '{conc_col}'")
            continue

        try:
            low_vals = pd.to_numeric(low_data[analyte], errors='coerce')
            conc_vals = pd.to_numeric(low_data[conc_col], errors='coerce')
            valid = low_vals.notna() & conc_vals.notna()

            if valid.sum() < 2:
                continue

            # --- Calculate SD and Slope ---
            low_sd = round(low_vals[valid].std(), 5)
            slope = round(((low_vals[valid].cov(conc_vals[valid])) / conc_vals[valid].var()), 5)

            if slope == 0:
                st.warning(f"⚠️ Slope is 0 for {analyte}. Cannot compute LOD.")
                continue

            lod = round((3.3 * low_sd) / slope, 5)
            loq = round(10 * low_sd, 5)

            results['Analyte'].append(analyte)
            results['Low SD'].append(low_sd)
            results['Slope'].append(slope)
            results['LOD'].append(lod)
            results['LOQ'].append(loq)

        except Exception as e:
            st.error(f"Error processing {analyte}: {e}")

    result_df = pd.DataFrame(results)
    st.subheader("📊 Limit of Detection (LOD) & Limit of Quantification (LOQ) Summary")
    st.dataframe(result_df)

    # st.download_button(
    #     label="⬇️ Download LOD/LOQ Results as CSV",
    #     data=result_df.to_csv(index=False),
    #     file_name="lod_loq_results.csv",
    #     mime="text/csv"
    # )

# --- Upload data ---
df = upload_data()  # Ensure the file is uploaded before performing analysis

# --- LOB Calculation Expander ---
with st.expander("📊 Calculate Limit of Blank (LOB)", expanded=True):
    if df is not None:
        calculate_lob(df)

# --- LOD & LOQ Calculation Expander ---
with st.expander("📊 Calculate Limit of Detection (LOD) & Limit of Quantification (LOQ)", expanded=True):
    if df is not None:
        calculate_lod_loq(df)

# --- Optional Reference Section ---
with st.expander("📚 References"):
    st.markdown("""
    **Armbruster, D.A. and Pry, T. (2008)**, *Limit of Blank, Limit of Detection and Limit of Quantitation*, The Clinical biochemist. Reviews, 29 Suppl 1(Suppl 1), S49–S52
    (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2556583/)
    """)
    st.markdown("""
    **Pum, J. (2019)**, *Chapter Six - A practical guide to validation and verification of analytical methods in the clinical laboratory*, Advances in Clinical Chemistry: Volume 90, pp. 215-281.
    (https://www.sciencedirect.com/science/article/pii/S006524231930006X)
    """)
