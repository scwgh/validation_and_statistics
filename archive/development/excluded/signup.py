if "show_popup" not in st.session_state:
    st.session_state.show_popup = True

if st.session_state.show_popup:
    with st.expander("📬 Sign up for email updates", expanded=True):
        name = st.text_input("Your Name")
        email = st.text_input("Email")
        if st.button("Sign Up"):
            if name and email:
                save_signup(name, email)
                st.success("Thank you! You're on the list.")
                st.session_state.show_popup = False
            else:
                st.warning("Please enter both fields.")


# --- ADMIN ONLY: Show email signups and CSV download (private) ---
with st.expander("👥 View Email Signups (Admin Only)", expanded=False):
    password = st.text_input("Admin password", type="password")
    if password:
        if password == admin_password:
            if st.button("Show Signups"):
                data = get_signups()
                if data:
                    df = pd.DataFrame(data, columns=["Name", "Email"])
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Signups as CSV", csv, "signups.csv", "text/csv")
                else:
                    st.info("No signups yet.")
        else:
            st.error("Incorrect password.")