import streamlit as st
import pandas as pd
import numpy as np
import os
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations
from datetime import datetime

def apply_app_styling():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
def grubbs_test(values: pd.Series, alpha: float = 0.05):
    outlier_indices = []
    values = values.copy()

    while len(values) > 2:
        n = len(values)
        mean = values.mean()
        std = values.std(ddof=1)

        abs_diff = np.abs(values - mean)
        max_diff_idx = abs_diff.idxmax()
        G = abs_diff[max_diff_idx] / std

        # Calculate critical value
        t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
        G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))

        if G > G_crit:
            outlier_indices.append(max_diff_idx)
            values = values.drop(index=max_diff_idx)
        else:
            break

    return {
        "Outlier Indices": outlier_indices,
        "Outliers": outlier_indices  # just index list for now, could add values if needed
    }



def check_westgard_rules(values, mean, sd, rules_enabled):
    alerts = []
    n = len(values)
    
    for i in range(n):
        val = values[i]
        z = (val - mean) / sd if sd != 0 else 0

        # 1_2s
        if rules_enabled['1_2s'] and abs(z) > 2:
            alerts.append((i, "1_2s"))

        # 1_3s
        if rules_enabled['1_3s'] and abs(z) > 3:
            alerts.append((i, "1_3s"))

        # 2_2s
        if rules_enabled['2_2s'] and i >= 1:
            prev_z = (values[i-1] - mean) / sd
            if abs(z) > 2 and abs(prev_z) > 2 and np.sign(z) == np.sign(prev_z):
                alerts.append((i-1, "2_2s"))
                alerts.append((i, "2_2s"))

        # R_4s
        if rules_enabled['R_4s'] and i >= 1:
            prev_z = (values[i-1] - mean) / sd
            if (z - prev_z) > 4:
                alerts.append((i-1, "R_4s"))
                alerts.append((i, "R_4s"))

        # 4_1s
        if rules_enabled['4_1s'] and i >= 3:
            zs = [(values[j] - mean) / sd for j in range(i-3, i+1)]
            if all(abs(zj) > 1 and np.sign(zj) == np.sign(zs[0]) for zj in zs):
                alerts.extend([(j, "4_1s") for j in range(i-3, i+1)])

        # 10x
        if rules_enabled['10x'] and i >= 9:
            zs = [(values[j] - mean) / sd for j in range(i-9, i+1)]
            if all(np.sign(zj) == np.sign(zs[0]) for zj in zs):
                alerts.extend([(j, "10x") for j in range(i-9, i+1)])

    return list(set(alerts))

def show_footer():
    st.markdown(
        """
        <footer>
            <p style='text-align: center; font-size: 14px;'>
                &copy; 2025 Validation and Statistical Analysis App. All rights reserved.
                <br>
                Licensed under the <a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank" style='color: #4C9ED9;'>Apache 2.0 License</a>. 
                This app does not store uploaded data and is intended for research and validation use only.
                No warranties are provided. The creators are not responsible for any results, interpretations, or decisions made using this app. You are responsible for ensuring compliance with local data protection and regulatory guidelines when uploading or analyzing data.
                <br><br>
                <a href="https://github.com/scwgh/validation_and_statistics" target="_blank" style='text-decoration: none; color: #4C9ED9;'>GitHub Repository</a> |
                <a href="mailto:validation.and.stats@gmail.com" style='text-decoration: none; color: #4C9ED9;'>Contact Support</a>
            </p>
        </footer>
        """,
        unsafe_allow_html=True
    )


