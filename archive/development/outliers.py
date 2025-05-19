import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from utils import apply_app_styling  # Assuming this is a custom function for styling

# Set Streamlit page configuration (move this to the top)
st.set_page_config(
    page_title="Outliers",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Grubbs' test function
def run_grubbs_test(data: pd.Series, alpha=0.05):
    data = data.dropna()
    n = len(data)
    if n < 3:
        return {"error": "Grubbs' test requires at least 3 data points."}

    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    abs_diff = np.abs(data - mean)
    max_diff_idx = abs_diff.idxmax()
    G = abs_diff[max_diff_idx] / std

    t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))

    result = {
        "N": n,
        "Mean": round(mean, 2),
        "SD": round(std, 2),
        "Max |x - mean|": round(abs_diff[max_diff_idx], 2),
        "G": round(G, 2),
        "G Critical": round(G_crit, 2),
        "Outlier Detected": G > G_crit,
        "Outlier Value": round(data[max_diff_idx], 2) if G > G_crit else None,
        "Outlier Index": round(max_diff_idx, 2) if G > G_crit else None,
    }

    return result

# Tietjen-Moore test function
def tietjen_moore_test(data: pd.Series, k: int):
    data = data.dropna()
    n = len(data)
    if k >= n or k <= 0:
        return {"error": "k must be greater than 0 and less than the number of observations."}
    if n < 3:
        return {"error": "Tietjen-Moore test requires at least 3 data points."}

    x = np.array(data)
    mean_x = np.mean(x)
    sorted_indices = np.argsort(np.abs(x - mean_x))
    
    trimmed = x[sorted_indices[:-k]]
    mean_trimmed = np.mean(trimmed)

    numerator = np.sum((trimmed - mean_trimmed) ** 2)
    denominator = np.sum((x - mean_x) ** 2)

    E = numerator / denominator

    crit_value = 1 - (k / n)

    result = {
        "N": n,
        "Suspected Outliers (k)": round(k, 2),
        "Test Statistic (E)": round(E, 2),
        "Approx Critical Value": round(crit_value, 2),
        "Outliers Detected": E < crit_value
    }

    return result

# Apply custom styling (assuming `apply_app_styling` function exists)
apply_app_styling()

# Streamlit app configuration
st.title("🎯 Outliers")

with st.expander("📘 Why are we interested in outliers?", expanded=False):
    st.markdown("""
        An outlier is defined as an observation (or datapoint) which deviates markedly from other observations. 
        Identification of potential outliers is important for a number of reasons:
        \n - An outlier may indicate "bad data" (e.g., a transcription error, a sample mix-up, etc.)
        \n - Random variation in the data.
        \n Iglewicz and Hoaglin (1987) highlight the following three problems regarding outliers:
        \n - Outlier labelling - this flags potential outliers for further investigation.
        \n - Outlier accommodation - this uses robust statistical methods to reduce the influence of outliers on the analysis.
        \n - Outlier identification = this formally tests whether an observation or datapoint is an outlier.
    """)
# ---Choosing an outliers test ---
with st.expander("📘 Which Outliers Test?", expanded=False):
    st.markdown("""
        - **Grubbs' Test**: This test is used to detect a single outlier in a univariate dataset. It is based on the maximum deviation from the mean and is sensitive to the presence of outliers.
        - **Tietjen-Moore Test**: This test is used to detect multiple outliers in a dataset. It is based on the trimmed mean and is less sensitive to the presence of outliers than Grubbs' test.
    """)

# --- Instructions ---
with st.expander("📘 Instructions:", expanded=False): 
    st.markdown("""
    This tool allows you to assess **intra-well, intra-batch, and inter-batch imprecision** across different levels of control or patient materials.

    1. **Upload your CSV file** – it should contain repeated measurements for the same sample/material across different runs or days.
        

    2. Your file should include the following columns:
    
    - `Date` – can be in short or long date format
    - `Test` (e.g., Intra_Batch_Imprecision, Inter_Batch_Imprecision)
    - `Analyser` (e.g., Waters TQ-D 1, Waters TQ-D 2)
    - `Material` (e.g., QC1, QC2, QC3)
    - `Sample ID` (e.g., 12345, 67890)
    - `Batch ID` (e.g., Test_Batch_123)
    - One or more **analyte columns** – ensure consistent naming and avoid use of special characters. You DO NOT need to include units in the column names.
    

    3. After upload, the app will:
    - Group data by `Material`, `QC Level`, and `Analyte`
    - Calculate intra-well, intra-batch, and inter-batch imprecision (if data is available).
    - Generate summary statistics and visualizations for each analyte

    4. **Select the analyte** you want to analyze from the dropdown menu.
    - The app will filter the data accordingly and display the results.
    - Use the toggle in the sidebar to enable or disable choose which outliers test you want to use. """)

# File upload handling
with st.expander("📤 Upload Your CSV File", expanded=True):
    st.markdown("Upload a CSV containing your analyte data.")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
    else:
        df = None
        st.info("Awaiting file upload...")

# Proceed with analysis if file is uploaded
if df is not None:
    required_columns = ['Material', 'Analyser', 'Sample ID']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
    else:
        st.subheader("🧪 Choose a Test to Run")

        # Select numeric columns for analysis (starting from 7th column)
        numeric_columns = df.columns[6:]
        if numeric_columns.empty:
            st.warning("No numeric columns found. Please upload appropriate data.")
        else:
            # User selects analyte and material
            selected_analyte = st.selectbox("Select 'Analyte' to test:", numeric_columns)
            selected_material = st.selectbox("Select 'Material' to test:", df['Material'].unique())

            filtered_df = df[df['Material'] == selected_material]

            # Tabs for Grubbs' Test and Tietjen-Moore Test
            tab1, tab2 = st.tabs(["Grubbs' Test", "Tietjen-Moore Test"])

            # Grubbs' Test
            with tab1:
                result = run_grubbs_test(filtered_df[selected_analyte])
                if "error" in result:
                    st.error(result["error"])
                else:
                    # Display Grubbs' test results
                    grubbs_table = {
                        "Test Statistic": ["N", "Mean", "SD", "Max |x - mean|", "G", "G Critical", "Outlier Detected"],
                        "Values": [result['N'], result['Mean'], result['SD'], result['Max |x - mean|'], result['G'], result['G Critical'], result['Outlier Detected']]
                    }
                    grubbs_df = pd.DataFrame(grubbs_table)
                    grubbs_df = grubbs_df.style.format({'Values': "{:.2f}", 'Outlier Detected': lambda x: 'Yes' if x else 'No'})
                    st.dataframe(grubbs_df)

                    import pandas as pd

                    # Check if an outlier is detected
                    if result['Outlier Detected']:
                        # Get the index of the detected outlier
                        outlier_index = int(result['Outlier Index'])
                        
                        # Retrieve the outlier row from the dataframe
                        outlier_row = filtered_df.iloc[outlier_index]
                        
                        # Prepare data for the table
                        outlier_data = {
                            "Analyte": [selected_analyte],
                            "Material": [outlier_row['Material']],                            
                            "Sample ID": [outlier_row['Sample ID']],                            
                            "Outlier Value": [result['Outlier Value']],
                            "Outlier Index": [outlier_index]
                        }
                        
                        # Create a DataFrame to display in table format
                        outlier_df = pd.DataFrame(outlier_data)
                        
                        # Display the outlier details in a table
                        st.write(f"### **Outlier Details:**")
                        st.dataframe(outlier_df)


            # Tietjen-Moore Test
            with tab2:
                k = st.number_input("Number of suspected outliers (k)", min_value=1, max_value=len(filtered_df)-1, value=1, step=1)
                result = tietjen_moore_test(filtered_df[selected_analyte], k)
                if "error" in result:
                    st.error(result["error"])
                else:
                    # Display Tietjen-Moore test results
                    tietjen_table = {
                        "Test Statistic": ["N", "Suspected Outliers (k)", "Test Statistic (E)", "Approx Critical Value", "Outliers Detected"],
                        "Values": [result['N'], result['Suspected Outliers (k)'], result['Test Statistic (E)'], result['Approx Critical Value'], result['Outliers Detected']]
                    }
                    tietjen_df = pd.DataFrame(tietjen_table)
                    tietjen_df = tietjen_df.style.format({'Values': "{:.2f}", 'Outliers Detected': lambda x: 'Yes' if x else 'No'})
                    st.dataframe(tietjen_df)

                    import pandas as pd

                    # Check if outliers are detected
                    if result['Outliers Detected']:
                        # Get the outlier indices based on sorted absolute deviations
                        outlier_indices = np.argsort(np.abs(filtered_df[selected_analyte] - np.mean(filtered_df[selected_analyte])))[:k]
                        
                        # Prepare data for the table
                        outliers_data = []
                        
                        for idx in outlier_indices:
                            outlier_row = filtered_df.iloc[idx]
                            outliers_data.append({
                                "Analyte": selected_analyte,
                                "Material": outlier_row['Material'],
                                "Sample ID": outlier_row['Sample ID'],
                                "Outlier Value": filtered_df[selected_analyte].iloc[idx]
                            })
                        
                        # Create a DataFrame to display in table format
                        outliers_df = pd.DataFrame(outliers_data)
                        
                        # Display the outlier details in a table
                        st.write(f"### **Outlier Indices**:")
                        st.dataframe(outliers_df)

with st.expander("‼️ Want to know more about outliers?!", expanded=False):  
    st.markdown("""
    \n If you want to think more about how outliers can affect outcomes, the **Monte Carlo simulation** is a model which predicts the probability of various outcomes when the potential for a random variable exists.
    
    It is not particularly relevant for outlier detection in a laboratory setting, but it is a useful tool for understanding the potential impact of outliers on the results of a statistical analysis.
    
    The method was developed by Jon von Neumann and Stanislaw Ulam during World War II to improve decision making in uncertain conditions. Unlike a normal forecasting model, the Monte Carlo simulation predicts a set of outcomes based on an estimate range of values versus a set input of fixed values. So, it builds a model of possible results by leveraging a probability distribution of the input variables. It then recalculates results over and over, each time using a different set of random values between the minimum and maximum values.
    
    When a Monte Carlo Simulation is complete, it yields a range of possible outcomes with the probability of each result occurring.

    One simple example of a Monte Carlo Simulation is to consider calculating the probability of rolling two standard dice. There are 36 combinations of dice rolls. Based on this, you can manually compute the probability of a particular outcome. Using a Monte Carlo Simulation, you can simulate rolling the dice 10,000 times (or more) to achieve more accurate predictions. 

    Using this method, you can more clearly identify datapoints which deviate from the expected pattern.
    """)

    # Monte Carlo Simulation Parameters
    n_simulations = 10000
    sample_size = 30
    means_normal = []
    means_with_outliers = []

    for _ in range(n_simulations):
        sample = np.random.normal(loc=100, scale=10, size=sample_size)
        means_normal.append(np.mean(sample))

        # Add an outlier to one random point
        sample_with_outlier = sample.copy()
        sample_with_outlier[np.random.randint(0, sample_size)] += np.random.choice([30, -30])
        means_with_outliers.append(np.mean(sample_with_outlier))

    # Create interactive Plotly histogram
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=means_normal,
        nbinsx=50,
        name='Normal Samples',
        opacity=0.6,
        marker_color='skyblue'
    ))

    fig.add_trace(go.Histogram(
        x=means_with_outliers,
        nbinsx=50,
        name='Samples with Outliers',
        opacity=0.6,
        marker_color='salmon'
    ))

    fig.update_layout(
        title="Monte Carlo Simulation: Effect of Outliers on Sample Mean",
        xaxis_title="Sample Mean",
        yaxis_title="Frequency",
        barmode='overlay',
        template='plotly_white',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0)', borderwidth=0)
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Optional Reference Section ---
with st.expander("📚 References"):
    st.markdown("""
    **Hoaglin, D.A. and Iglewicz, B.** (1987). Fine-Tuning Some Resistant Rules for Outlier Labeling. Journal of the American Statistical Association, 82(400), 1147–1149. DOI: https://doi.org/10.2307/2289392
    \n **Raychaudhuri, S.**  (2008), Introduction to Monte Carlo simulation, 2008 Winter Simulation Conference, Miami, FL, USA, pp. 91–100. DOI: 10.1109/WSC.2008.4736059.""")