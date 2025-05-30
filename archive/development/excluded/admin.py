import streamlit as st
from signup_utils import get_all_signups
import pandas as pd

PASSWORD = "your_secret_admin_password"

st.title("Admin Panel")

password = st.text_input("Enter admin password", type="password")
if password == PASSWORD:
    data = get_all_signups()
    df = pd.DataFrame(data, columns=["Name", "Email", "Timestamp"])
    st.dataframe(df)
    
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, "signups.csv", "text/csv")
else:
    st.warning("Unauthorized. Please enter the correct password.")
