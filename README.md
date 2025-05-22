from pathlib import Path

# 🧪 Validation and Statistical Analysis App (Beta)

A user-friendly, Streamlit-powered web application to support laboratory professionals in conducting validation and verification analyses using a suite of statistical tools.

---

## 📋 Overview

This app provides interactive tools and visualizations for performing statistical analyses common in laboratory method validation. Built for flexibility and ease of use, it supports a wide range of tests including precision, linearity, limits of detection/quantitation, and method comparison.

---

## 🚀 Features

### 🔬 Validation Modules
- **Imprecision Analysis**
  - Intra-Well
  - Intra-Batch
  - Inter-Batch
  - Total Imprecision
- **Linearity**
  - Standard Curve
  - Response Curve
- **Detection & Quantitation**
  - Limit of Detection (LOD)
  - Limit of Quantitation (LOQ)
- **Method Comparison**
  - ANOVA
  - Bland-Altman
  - Deming Regression
  - Passing-Bablok
- **Additional Statistical Tests**
  - Chi-Squared, F-test, Levene’s Test
  - Mann-Whitney U, Kruskal-Wallis
  - Shapiro-Wilk, Z-test, T-test
  - P-P & Q-Q Plots
  - TEa (Total Allowable Error)
  - and more...

---

## 🗂️ Data Upload & Format

- Input data should be in **CSV format**.
- Required columns include: `Date`, `Material`, `Analyser`, and `Sample ID`.
- Analyte columns should be named clearly without **spaces or special characters** (`@ : ; ! , . # < >`).
- Each module provides downloadable templates to ensure correct formatting.

---

## 📦 Dependencies

This app uses the following core libraries:

- `streamlit`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`, `plotly`
- `scipy`, `statsmodels`
- `kaleido` (for exporting plots)

- Custom styling via `utils.py`

## Privacy Policy

See [PRIVACY.md](./PRIVACY.md) for details on how data is handled.


## ▶️ Getting Started

To run the app locally:

```bash
# Clone the repository
git clone https://github.com/your-repo/validation-analysis-app.git
cd validation-analysis-app

# Install dependencies (recommended to use a virtual environment)
pip install -r requirements.txt

# Run the Streamlit app
streamlit run main.py
