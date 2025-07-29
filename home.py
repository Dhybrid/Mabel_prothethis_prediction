import streamlit as st
from PIL import Image
import requests
from io import BytesIO

def show():
    # Title
    st.markdown(
        "<h1 style='text-align:center; color:#3366cc;'>🦿 Osteoarthritis & Fracture Predictor</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; font-size:18px; color:#7f8c8d;'>Empowering bone health decisions for prosthetic users and the elderly</p>",
        unsafe_allow_html=True
    )

    st.divider()

    # Load and Show Image
    try:
        url = "https://images.unsplash.com/photo-1611782373786-2c723b1b531b?auto=format&fit=crop&w=800&q=80"
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
    except:
        img = Image.open("assets/knee.webp")

    st.image(img, caption="🦴 Joint Health Matters", use_container_width=True)

    st.divider()

    # Copyright Notice
    with st.container():
        st.markdown("### 📜 Copyright Notice")
        st.markdown("""
        **© 2025 Babatunda & Mabel — All rights reserved.**

        This project was developed by **Babatunda** and **Mabel**,  
        students of the **Prosthetics Department** at  
        the **Federal University of Technology Owerri (FUTO)**.
        """)

    st.divider()

    # Beautiful About Section (No HTML, all clean text)
    with st.container():
        st.markdown("### 🧠 About This App")
        st.markdown("This predictive health tool is designed to assist with:")

        st.markdown("- **Osteoarthritis Detection** in prosthetic users — *by Babatunda*")
        st.markdown("- **Fragility Fracture Risk Assessment** for elderly individuals or caregivers — *by Mabel*")

        st.markdown("Click the button below to begin your personalized assessment.")

    st.write("")
    if st.button("🚀 Continue to Prediction", use_container_width=True):
        st.session_state.page = "predict"
        st.rerun()
