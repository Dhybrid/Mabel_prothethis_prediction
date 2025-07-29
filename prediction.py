import streamlit as st
import prediction_client
import prediction_partner

def show():
    st.title("🧠 Choose Prediction Type")
    st.markdown("Select the type of prediction you want to perform:")

    option = st.selectbox("Choose Model", ["-- Select --", "Mabel's Prediction", "Babatunda's Prediction"])

    if option == "Mabel's Prediction":
        prediction_client.show()
    elif option == "Babatunda's Prediction":
        prediction_partner.show()
