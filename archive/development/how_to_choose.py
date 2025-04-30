import streamlit as st
import pandas as pd
from utils import apply_app_styling

# Page setup
st.set_page_config(
    page_title="How to Choose",
    page_icon="♾️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_app_styling()
st.title("♾️ How to choose which statistical test is most suitable?")