import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Load model ===
@st.cache_resource
def load_model():
    return joblib.load("models/osteoporosis_risk_model.pkl")

model = load_model()


def show():
    # === App UI ===
    st.title("🦴 Osteoporosis Risk Prediction (Amputation Context)")
    st.markdown("Fill in the patient's clinical, demographic, and lifestyle details:")

    # === Collect Inputs ===
    age = st.slider("Age", 18, 100, 45)
    gender = st.selectbox("Gender", ["Female", "Male"])
    education = st.selectbox("Level of Education", ["Primary", "Secondary", "Tertiary"])
    ethnicity = st.selectbox("Ethnicity", [
        "Akwa ibom", "Akwaibom", "Benin", "Efik", "Hausa",
        "Ife", "Igbo", "Minority ethnic group", "Tiv", "Yoruba"
    ])
    weight = st.number_input("Weight (kg)", 30.0, 200.0, step=0.5)
    height = st.number_input("Height (m)", 1.0, 2.5, step=0.01)

    bmi = round(weight / (height ** 2), 2)
    st.markdown(f"**BMI:** `{bmi}`")

    bone_density = st.slider("Bone Density (T-score)", -4.0, 2.5, -1.0)

    # Binary/Int Inputs
    bool_map = {"Yes": 1, "No": 0}

    osteoporosis_fam = st.selectbox("Osteoporosis Family History", list(bool_map.keys()))
    osteoporosis_diag = st.selectbox("Diagnosed with Osteoporosis?", list(bool_map.keys()))
    osteopenia_fam = st.selectbox("Osteopenia Family History", list(bool_map.keys()))
    osteopenia_diag = st.selectbox("Diagnosed with Osteopenia?", list(bool_map.keys()))

    amp_type = st.slider("Type of Amputation (Numeric ID)", 0, 5, 2)
    amp_cause = st.slider("Cause of Amputation (Numeric ID)", 0, 5, 1)
    years_amp = st.slider("Years with Amputation", 0, 60, 6)
    glucocorticoids = st.selectbox("Glucocorticoid Use?", list(bool_map.keys()))
    chronic_illness = st.slider("Chronic Illness Count", 0, 5, 2)
    chronic_detail = st.slider("Chronic Illness Detail ID", 0, 5, 2)

    supplements = st.selectbox("Taking Bone Supplements?", list(bool_map.keys()))
    gait_diff = st.selectbox("Gait Difficulty?", list(bool_map.keys()))
    gait_change = st.selectbox("Change in Gait?", list(bool_map.keys()))
    assistive_device = st.selectbox("Using Assistive Device?", list(bool_map.keys()))
    daily_activities = st.selectbox("Independent in Daily Activities?", list(bool_map.keys()))
    rom_limit = st.selectbox("Range of Motion Limitations?", list(bool_map.keys()))
    activity_lim = st.selectbox("Activity Performance Change?", list(bool_map.keys()))
    pain_change = st.selectbox("Increased Pain Over Time?", list(bool_map.keys()))

    prosthesis_type = st.slider("Lower Limb Prosthesis Type (Numeric ID)", 0, 3, 2)
    history_falls = st.selectbox("History of Falls?", list(bool_map.keys()))
    prosthesis_years = st.slider("Years Using Prosthetic Limb", 0, 50, 3)
    prosthesis_issues = st.selectbox("Complications with Prosthesis?", list(bool_map.keys()))

    activity_level = st.selectbox("Activity Level with Prosthesis (ID)", [0, 1, 2])
    regular_exercise = st.selectbox("Engages in Regular Exercise?", list(bool_map.keys()))
    exercise_minutes = st.slider("Minutes of Exercise", 0.0, 120.0, 30.0)

    smoke = st.selectbox("Currently Smoke?", list(bool_map.keys()))
    smoke_past = st.selectbox("Smoked in the Past?", list(bool_map.keys()))
    tobacco_use = st.selectbox("Use Tobacco Substances?", list(bool_map.keys()))
    tobacco_past = st.selectbox("Used Tobacco in the Past?", list(bool_map.keys()))
    alcohol = st.selectbox("Consume Alcohol Regularly?", list(bool_map.keys()))

    # === Feature Dictionary ===
    input_dict = {
        'Age': age,
        'Gender': 1 if gender == "Male" else 0,
        'Level of education': ["Primary", "Secondary", "Tertiary"].index(education),
        'Ethnicity': [
            "Akwa ibom", "Akwaibom", "Benin", "Efik", "Hausa",
            "Ife", "Igbo", "Minority ethnic group", "Tiv", "Yoruba"
        ].index(ethnicity),
        'Weight': weight,
        'Height': height,
        'BMI': bmi,
        'Bone Density': bone_density,
        'Osteoporosis Family History': bool_map[osteoporosis_fam],
        'Have you been diagnosed with osteoporosis?': bool_map[osteoporosis_diag],
        'Osteopenia Family History': bool_map[osteopenia_fam],
        'Osteopenia Diagnosed': bool_map[osteopenia_diag],
        'What type of amputation?': amp_type,
        'What caused the amputation?': amp_cause,
        'For how long have you been with amputation?': years_amp,
        'Glucocorticoid Use': bool_map[glucocorticoids],
        'Chronic Illnesses': chronic_illness,
        'Chronic Conditions Detail': chronic_detail,
        'Are you taking any supplements(e.g. Calcium, Vitamin D) to support bone health?': bool_map[supplements],
        'Gait Difficulty': bool_map[gait_diff],
        'Have you noticed any changes in your gait or walking pattern?': bool_map[gait_change],
        'Assistive Device Used': bool_map[assistive_device],
        'Can you perform daily activities (e.g. Bathing, Dressing, Cooking) without difficulties?': bool_map[daily_activities],
        'Do you have any limitations in your range of motion or flexibility?': bool_map[rom_limit],
        'Have you noticed any changes in your ability to perform physical activities (e.g. Walking, Climbing stairs)?': bool_map[activity_lim],
        'Have you noticed any increase in your pain levels over time?': bool_map[pain_change],
        'What type of lower limb prosthesis do you use?': prosthesis_type,
        'Do you have a history of falls while using the prosthetic device?': bool_map[history_falls],
        'How long have you been using a prosthetic limb?': prosthesis_years,
        'Have you experienced any complications with your prosthetic limb (e.g. Pain, Instability, loose socket fitting, discomfort, misalignment)?': bool_map[prosthesis_issues],
        'What is your level of activity with the prosthesis?': activity_level,
        'Do you engage in regular exercise (long distance walks e.t.c)?': bool_map[regular_exercise],
        'If yes, for how many minutes do you exercise?': exercise_minutes,
        'Do you smoke?': bool_map[smoke],
        'Have you smoked in the past?': bool_map[smoke_past],
        'Do you take/use tobacco substances?': bool_map[tobacco_use],
        'Have you taken/used tobacco substances in the past?': bool_map[tobacco_past],
        'Do you consume alcohol regularly?': bool_map[alcohol],
        'Bone Density (T-score)': bone_density,
        'Activity Level': activity_level,
    }

    # # === Predict Button ===
    # if st.button("🔍 Predict Risk"):
    #     input_df = pd.DataFrame([input_dict])

    #     prediction = model.predict(input_df)[0]
    #     probability = model.predict_proba(input_df)[0][1]

    #     st.subheader("📊 Prediction Result")
    #     st.write(f"**Prediction:** {'At Risk' if prediction == 1 else 'No Risk'}")
    #     st.write(f"**Risk Probability:** `{probability:.3f}`")

    #     # Optional: Add guidance
    #     if probability >= 0.6:
    #         st.warning("⚠️ High Risk: Immediate attention required.")
    #     elif probability >= 0.4:
    #         st.info("🔎 Moderate Risk: Monitor and reassess.")
    #     else:
    #         st.success("✅ Low Risk: Maintain healthy habits.")

    import matplotlib.pyplot as plt  # Add this at the top of your file

    if st.button("🔍 Predict Risk"):
        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.toast("✅ Prediction Complete", icon="🧠")
        st.markdown("<h2>📋 <b>Result Summary</b></h2>", unsafe_allow_html=True)

        diagnosis_html = (
            "<h3>🧠 Diagnosis: <span style='color:red; font-weight:bold;'>At Risk of Osteoporosis</span></h3>"
            if prediction == 1 else
            "<h3>🧠 Diagnosis: <span style='color:green; font-weight:bold;'>No Risk Detected</span></h3>"
        )
        st.markdown(diagnosis_html, unsafe_allow_html=True)

        if probability >= 0.6:
            risk_level = "🔴 <b style='color:red;'>High</b>"
        elif probability >= 0.4:
            risk_level = "🟠 <b style='color:orange;'>Medium</b>"
        else:
            risk_level = "🔵 <b style='color:blue;'>Low</b>"

        st.markdown(f"""
        <div style='font-size:18px;'>
            📊 <b>Confidence Score:</b> {probability:.2%}<br>
            🧪 <b>Risk Level:</b> {risk_level}
        </div>
        """, unsafe_allow_html=True)

        # # Bone Mineral Density Chart
        # if 'bone_density' in input_dict and input_dict['bone_density'] not in [None, '', 'unknown']:
        #     st.markdown("### 📉 Bone Mineral Density (BMD) Overview")
        #     try:
        #         bone_density = float(input_dict['bone_density'])
        #         fig, ax = plt.subplots()
        #         ax.bar(["Your BMD"], [bone_density], color="green" if bone_density >= -1 else "red")
        #         ax.axhline(y=-1, color='blue', linestyle='--', label='Normal Threshold')
        #         ax.set_ylabel("BMD (T-score)")
        #         ax.set_title("Bone Density Level")
        #         ax.legend()
        #         st.pyplot(fig)
        #     except ValueError:
        #         st.warning("⚠️ Invalid bone density value provided.")
        # else:
        #     st.warning("🦴 Bone density not provided and could not be estimated. Provide it for more accurate results.")

        # Recommendations
        st.markdown("### 🧭 Personalized Recommendations")
        if prediction == 1:
            st.markdown("""
            - 🏥 **Consult a specialist** for further testing  
            - 💊 Consider **bone-strengthening medications** (e.g., bisphosphonates, denosumab)  
            - 🥗 Follow a **calcium & vitamin D-rich diet**  
            - 🏃‍♀️ Engage in **weight-bearing and resistance exercises**  
            - 🧘‍♂️ Practice **fall prevention strategies**  
            - 🩺 Schedule **regular bone density scans**
            """)
        else:
            st.markdown("""
            - ✅ Maintain **a healthy weight and lifestyle**  
            - 🥦 Consume foods rich in **calcium and vitamin D**  
            - 🚭 Avoid **smoking and alcohol**  
            - 🏃‍♂️ Stay active with **low-impact exercises**  
            - 🩺 Get routine **check-ups and scans**  
            - 🔍 Be alert for early signs (e.g., stiffness, mobility issues)
            """)

        # Possible Causes
        st.markdown("### 🧬 Possible Contributing Factors")
        contributing_factors = []

        if input_dict.get("glucocorticoid_use", "no").lower() == "yes":
            contributing_factors.append("Chronic glucocorticoid use")
        if input_dict.get("osteoporosis_family_history", "no").lower() == "yes":
            contributing_factors.append("Family history of osteoporosis")
        if input_dict.get("chronic_illnesses", "no").lower() == "yes":
            contributing_factors.append("Presence of chronic illness")
        if input_dict.get("supplements", "no").lower() == "no":
            contributing_factors.append("Lack of bone-supporting supplements")

        if contributing_factors:
            for factor in contributing_factors:
                st.markdown(f"- 🔸 {factor}")
        else:
            st.markdown("No major contributing risk factors detected in your input.")

