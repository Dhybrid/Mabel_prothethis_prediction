import streamlit as st
import home
import prediction

st.set_page_config(page_title="Fragility Fracture & Osteoarthritis Prediction", layout="centered")

# Initialize session state if not already set
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# If navigating from the home screen via button
if st.session_state.page == "predict":
    prediction.show()
else:
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Home", "🧠 Predict"])

    if page == "🏠 Home":
        home.show()
    elif page == "🧠 Predict":
        prediction.show()
