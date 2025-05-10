import streamlit as st
import importlib
from scipy import stats
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import apply_app_styling
import matplotlib.pyplot as plt
def run_grubbs_test(data: pd.Series, alpha=0.05):
    """
    Perform Grubbs' test to detect a single outlier in a univariate dataset.
    
    Parameters:
        data (pd.Series): A 1D pandas Series of numeric data.
        alpha (float): Significance level for the test.

    Returns:
        dict: A dictionary containing test results, including G statistic,
              critical value, and whether the minimum or maximum is an outlier.
    """
    data = data.dropna()
    n = len(data)
    if n < 3:
        return {"error": "Grubbs' test requires at least 3 data points."}

    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    abs_diff = np.abs(data - mean)
    max_diff_idx = abs_diff.idxmax()
    G = abs_diff[max_diff_idx] / std

    # Critical value
    t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))

    return {
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


# Tietjen-Moore Test Function
def tietjen_moore_test(data: pd.Series, k: int):
    """
    Perform the Tietjen-Moore test for detecting k outliers in a univariate dataset.
    
    Parameters:
        data (pd.Series): The input data as a pandas Series.
        k (int): Number of suspected outliers.

    Returns:
        dict: Test statistic, approximate critical value, and decision.
    """
    data = data.dropna()
    n = len(data)
    if k >= n or k <= 0:
        return {"error": "k must be greater than 0 and less than the number of observations."}
    if n < 3:
        return {"error": "Tietjen-Moore test requires at least 3 data points."}

    x = np.array(data)
    mean_x = np.mean(x)
    sorted_indices = np.argsort(np.abs(x - mean_x))
    
    # Remove k most extreme values
    trimmed = x[sorted_indices[:-k]]
    mean_trimmed = np.mean(trimmed)

    numerator = np.sum((trimmed - mean_trimmed) ** 2)
    denominator = np.sum((x - mean_x) ** 2)

    E = numerator / denominator

    # Approximate critical value
    crit_value = 1 - (k / n)

    return {
        "N": n,
        "Suspected Outliers (k)": round(k, 2),
        "Test Statistic (E)": round(E, 2),
        "Approx Critical Value": round(crit_value, 2),
        "Critical Value": round(1 - (k / n), 2),
        "Outliers Detected": E < crit_value
    }
# --- Streamlit App ---
# Page setup
st.set_page_config(
    page_title="Outliers",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_app_styling()

st.title("🎯 Outliers")

with st.expander("📘 Why are we interested in outliers?", expanded=True):
    st.markdown("""An outlier is defined as an observation (or datapoint) which deviates markedly from other observations. Identification of potential outliers is important for a number of reasons:
                \n - An outlier may indicate "bad data" (e.g., a transcription error, a sample mix-up, etc.)
                \n - Random variation in the data. Random variation may happen and we can continue with our tests without worrying too much about this - but this not mean that we will want to exclude the outlying result from our data. 
                \n Iglewicz and Hoaglin (1987) highlight the following three problems regarding outliers:
                \n - Outlier labelling - this flags potential outliers for further investigation. 
                \n - Outlier accomomodation - this uses robust statistical methods to reduce the influence of outliers on the analysis.
                \n - Outlier identification = this formally tests whether an observation or datapoint is an outlier. 
    """)
with st.expander("📘 What is masking and swamping?", expanded=False):
    st.markdown("""Masking occurs when an outlier is not detected because it is masked by other outliers - or "hidden". For example, if a dataset contains two outliers, one of which is much larger than the other, the larger outlier may mask the smaller one, making it difficult to identify. This can lead to incorrect conclusions about the data and can affect the results of statistical analyses.
    \n Masking can occur in various statistical tests and methods, including regression analysis, ANOVA, and clustering algorithms. It is important to be aware of this phenomenon and to use appropriate techniques to detect and handle outliers in your data.
    """)
    st.markdown("""Swamping occurs when an outlier is not detected because it is swamped by too many other data points. For example, if a dataset contains a large number of data points that are very close together, an outlier that is far away from the rest of the data may be swamped by the other points, making it difficult to identify. This can lead to incorrect conclusions about the data and can affect the results of statistical analyses.
    \n Masking and swamping can occur in various statistical tests and methods, including regression analysis, ANOVA, and clustering algorithms. It is important to be aware of this phenomenon and to use appropriate techniques to detect and handle outliers in your data.
    """)
with st.expander("📘 What are Z-Scores and Modified Z-scores?", expanded=False):
    st.markdown(""" The Z-score of an observation is defined as:""")
    st.latex(r''' \zeta = \frac{x - \mu}{\sigma}''')
    st.markdown("""whereby $\mu$ and $\sigma$ denote the sample mean and standard deviation respectively. Although it is common practice to use Z-scores to identify possible outliers, this can be misleading as the maximum Z-score which might be observed is ${(n - 1)}/{\sqrt{n}}$.""")
    st.markdown("""The modified Z-score is defined as:""") 
    st.latex(r''' {M}_i = \frac{0.6745(x - \mu)}{MAD}''')
    st.markdown("""whereby MAD is the median absolute deviation. The modified Z-score is less sensitive to outliers than the standard Z-score, making it a more robust measure of how far an observation is from the mean of the data set. The modified Z-score is often used in robust statistics and is particularly useful for identifying outliers in small data sets or data sets with a high degree of variability.""")

with st.expander("📘 How do I test for outliers?", expanded=False):
    st.markdown("""There are two established tests for identification of outliers: Grubbs' test and the Tietjen-Moore test. A generalised extreme studentized deviate test (ESD) is also available. 
                \n - **The Tietjen-Moore test** is a non-parametric test for outliers that is based on the ranks of the data. It is particularly useful when the data are not normally distributed or when the sample size is small. The Tietjen-Moore test is less sensitive to outliers than other tests, such as Grubbs' test, and can be used to identify both univariate and multivariate outliers.
                \n - **The Grubbs' test** is used to detect outliers in a univariate data set. It is based on the assumption that the data are normally distributed and that the outliers are extreme values that are significantly different from the rest of the data. """)
with st.expander("📘 Instructions:"):
    st.markdown("""
    1. Upload a CSV file containing your data.
    2. Ensure the file includes columns for `Analyser`, `Material`, `Sample ID`, and 'Analyte Name'.
    3. Select which test you want to perform.
    4. Click the button below to run the outlier analysis.
    """)

# File upload and preview
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

# Check if file is uploaded and proceed with the analysis
if df is not None:
    # Verify required columns exist in the dataset
    required_columns = ['Material', 'Analyser', 'Sample ID']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
    else:
        st.subheader("🧪 Choose a Test to Run")

        # Select numeric columns for analysis
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_columns:
            st.warning("No numeric columns found. Please upload appropriate data.")
        else:
            # Allow user to select analyte and material for testing
            selected_analyte = st.selectbox("Select 'Analyte' to test:", numeric_columns)
            selected_material = st.selectbox("Select 'Material' to test:", df['Material'].unique())

            # Filter the dataframe for the selected analyte and material
            filtered_df = df[(df['Material'] == selected_material)]

            # Test buttons layout
            test_col1, test_col2 = st.columns(2)

            with test_col1:
                if st.button("Run Grubbs' Test"):
                    result = run_grubbs_test(filtered_df[selected_analyte])
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.write("**Grubbs' Test Result:**")
                        st.write(f"N: {result['N']}")
                        st.write(f"Mean: {result['Mean']}")
                        st.write(f"SD: {result['SD']}")
                        st.write(f"Max |x - mean|: {result['Max |x - mean|']}")
                        st.write(f"G: {result['G']}")
                        st.write(f"G Critical: {result['G Critical']}")
                        st.write(f"Outlier Detected: {result['Outlier Detected']}")
                        if result['Outlier Detected']:
                            st.write(f"Outlier Value: {result['Outlier Value']}")
                            st.write(f"Outlier Index: {result['Outlier Index']}")

            with test_col2:
                k = st.number_input("Number of suspected outliers (k)", min_value=1, max_value=len(filtered_df)-1, value=1, step=1)
                if st.button("Run Tietjen-Moore Test"):
                    result = tietjen_moore_test(filtered_df[selected_analyte], k)
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.write("**Tietjen-Moore Test Result:**")
                        st.write(f"N: {result['N']}")
                        st.write(f"Suspected Outliers (k): {result['Suspected Outliers (k)']}")
                        st.write(f"Test Statistic (E): {result['Test Statistic (E)']}")
                        st.write(f"Approx Critical Value: {result['Approx Critical Value']}")
                        st.write(f"Outliers Detected: {result['Outliers Detected']}")

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
    **Hoaglin, D.A. and Iglewicz, B.** (1987). Fine-Tuning Some Resistant Rules for Outlier Labeling. Journal of the American Statistical Association, 82(400), 1147–1149. https://doi.org/10.2307/2289392
    """)