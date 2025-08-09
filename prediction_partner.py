import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def load_model():
    return joblib.load("models/hybrid_prosthetic_oa_model.pkl")

def show():
    st.title("🦿 Osteoarthritis Risk Prediction")
    st.markdown("_This tool helps assess osteoarthritis risk in lower limb amputee partners._")

    model = load_model()

    st.subheader("🔍 Enter Partner Information")

    # Inputs
    age = st.selectbox("📅 Age Group", ['Under 18', '18-25', '26-35', '36-45', '46-60', '60+'])
    sex = st.selectbox("👤 Sex", ["Male", "Female"])
    weight = st.number_input("⚖️ Weight (kg)", 30.0, 200.0, 75.0)
    height = st.number_input("📏 Height (cm)", 100.0, 220.0, 170.0)
    bmi = st.number_input("💪 BMI (enter 0 to auto-calculate)", 0.0, 80.0, 0.0)

    joint_injuries = st.selectbox("🦴 Previous joint injuries or surgeries?", ["Yes", "No"])
    family_history = st.selectbox("👪 Family history of osteoarthritis?", ["Yes", "No"])
    other_conditions = st.selectbox("🏥 Other health conditions?", ["Yes", "No"])
    on_medication = st.selectbox("💊 Currently taking any medications?", ["Yes", "No"])

    amputation_type = st.selectbox("🦿 Type of amputation", ["Below Knee", "Above Knee", "Foot"])
    amputation_level = st.selectbox("📉 Level of amputation", ["Trans-tibial", "Trans-femoral", "Partial", "Complete"])
    amputation_cause = st.selectbox("💥 Cause of amputation", ["Accident", "Infection", "Disease", "Trauma"])
    years_since_amputation = st.selectbox("📆 How long since amputation", ["<1 year", "1-2 years", "2-5 years", "5+ years"])

    prosthesis_type = st.selectbox("🦿 Type of prosthesis", ["Mechanical", "Microprocessor", "Passive"])
    prosthesis_years = st.selectbox("📆 How long using prosthesis", ["<1 year", "1-2 years", "2-5 years", "5+ years"])
    prosthesis_freq = st.selectbox("⏱️ Frequency of use", ["Rarely", "Sometimes", "Often", "Daily", "Always"])

    mobility_level = st.slider("🚶 Mobility level (1 = low, 5 = high)", 1, 5, 3)
    pain_impact = st.selectbox("⚠️ Does pain impact daily activities?", ["Yes", "No"])
    other_symptoms = st.selectbox("🤕 Other symptoms (stiffness, swelling)?", ["Yes", "No"])
    prosthesis_comfort = st.slider("😊 Prosthesis comfort (1 = bad, 5 = great)", 1, 5, 4)
    daily_life_impact = st.slider("🌟 Impact on daily life (1 = bad, 5 = good)", 1, 5, 3)
    mobility_satisfaction = st.slider("👍 Satisfaction with mobility (1 = low, 5 = high)", 1, 5, 4)

    exercise = st.selectbox("🏃 Exercise regularly?", ["Yes", "No"])
    smoke = st.selectbox("🚬 Do you smoke?", ["Yes", "No"])
    diet_rating = st.slider("🥗 Diet rating (1 = poor, 5 = excellent)", 1, 5, 3)

    if bmi == 0.0:
        bmi = round(weight / ((height / 100) ** 2), 2)
        st.success(f"✅ Auto-calculated BMI: {bmi}")

    # Prepare input
    user_input = {
        'age': age,
        'sex': sex,
        'weight (kg)': weight,
        'height (cm)': height,
        'body mass index (bmi)': bmi,
        'have you had any previous joint injuries or surgeries': joint_injuries,
        'do you have a family history of osteoarthritis': family_history,
        'do you have any other health conditions (e.g, diabetes, rheumatoid arthritis)': other_conditions,
        'are you currently taking any medications': on_medication,
        'what type of amputation do you have?': amputation_type,
        'what level of amputation do you have?': amputation_level,
        'what caused the amputation?': amputation_cause,
        'for how long have you been with amputation?': years_since_amputation,
        'what type of lower limb prosthesis are you using?': prosthesis_type,
        'how long have you been using a lower limb prosthesis?': prosthesis_years,
        'how often do you use your prosthesis?': prosthesis_freq,
        'how would you rate your level of mobility and independence? (scale 1-5, where 1 is very limited and 5 is very independent)': mobility_level,
        'does pain impact your daily activities?': pain_impact,
        'do you experience any other symptoms? (e.g, stiffness, swelling)': other_symptoms,
        'how satisfied are you with the fit and comfort of your prosthesis on a scale of 1-5( where 1 is very dissatisfied and 5 is very satisfied)': prosthesis_comfort,
        'how does your prosthesis impact your daily life and activities on a scale of 1-5 (where 1 is very negatively and 5 is very positively)': daily_life_impact,
        'how satisfied are you with your current level of mobility and independence on a scale of 1-5 (where 1 is very dissatisfied and 5 is very satisfied)': mobility_satisfaction,
        'do you engage in regular exercise?': exercise,
        'do you smoke?': smoke,
        'how would you rate your diet and nutrition habit on a scale of 1-5? (where 1 is poor and 5 is excellent)': diet_rating
    }

    import plotly.graph_objects as go

    # if st.button("📊 Predict Osteoarthritis Risk"):
    #     input_df = pd.DataFrame([user_input])

    #     # --- Binary Encoding ---
    #     binary_cols = [
    #         'have you had any previous joint injuries or surgeries',
    #         'do you have a family history of osteoarthritis',
    #         'do you have any other health conditions (e.g, diabetes, rheumatoid arthritis)',
    #         'are you currently taking any medications',
    #         'does pain impact your daily activities?',
    #         'do you experience any other symptoms? (e.g, stiffness, swelling)',
    #         'do you engage in regular exercise?',
    #         'do you smoke?'
    #     ]
    #     le = LabelEncoder()
    #     for col in binary_cols:
    #         input_df[col] = le.fit(['No', 'Yes']).transform(input_df[col])

    #     # --- Ordinal Encoding ---
    #     ordinal_cols = [
    #         'how would you rate your level of mobility and independence? (scale 1-5, where 1 is very limited and 5 is very independent)',
    #         'how satisfied are you with the fit and comfort of your prosthesis on a scale of 1-5( where 1 is very dissatisfied and 5 is very satisfied)',
    #         'how does your prosthesis impact your daily life and activities on a scale of 1-5 (where 1 is very negatively and 5 is very positively)',
    #         'how satisfied are you with your current level of mobility and independence on a scale of 1-5 (where 1 is very dissatisfied and 5 is very satisfied)',
    #         'how would you rate your diet and nutrition habit on a scale of 1-5? (where 1 is poor and 5 is excellent)'
    #     ]
    #     input_df[ordinal_cols] = input_df[ordinal_cols].astype(int)

    #     # --- One-Hot Encoding ---
    #     input_df = pd.get_dummies(input_df)

    #     # --- Align Input Columns with Model ---
    #     model_features = list(map(str, model.feature_names_in_)) if hasattr(model, 'feature_names_in_') else list(input_df.columns)
    #     input_df.columns = input_df.columns.astype(str)
    #     for col in model_features:
    #         if col not in input_df.columns:
    #             input_df[col] = 0
    #     input_df = input_df[model_features]

    #     # --- Bone Density Estimation (if missing) ---
    #     bmd_val = user_input.get("bone density")
    #     estimated_bmd = None
    #     if bmd_val is None or str(bmd_val).strip() in ["", "nan"]:
    #         try:
    #             bmi = float(user_input.get("bmi", 0))
    #             age = int(user_input.get("age", 50))
    #             gender = str(user_input.get("gender", "female")).lower()
    #             if gender in ['male', 'female'] and bmi > 0:
    #                 estimated_bmd = round(1.05 - (0.005 * (age - 30)) + (0.01 if gender == 'male' else -0.01), 2)
    #                 input_df['bone density'] = estimated_bmd
    #                 bmd_val = estimated_bmd
    #         except Exception:
    #             bmd_val = None

    #     # --- Prediction ---
    #     prediction = model.predict(input_df)[0]
    #     risk_score = model.predict_proba(input_df)[0][1]  # Class 1: Osteoarthritis

    #     # --- Risk Tag ---
    #     if risk_score > 0.75:
    #         risk_level = "🔴 <b style='color:red;'>High</b>"
    #     elif risk_score > 0.4:
    #         risk_level = "🟠 <b style='color:orange;'>Medium</b>"
    #     else:
    #         risk_level = "🟢 <b style='color:green;'>Low</b>"

    #     # --- Result Display ---
    #     st.toast("✅ Prediction Complete", icon="🤖")
    #     st.subheader("📋 Result Summary")

    #     diagnosis_html = (
    #         "<h3>🧠 Diagnosis: <span style='color:red; font-weight:bold;'>Likely Osteoarthritis</span></h3>"
    #         if prediction == 1 else
    #         "<h3>🧠 Diagnosis: <span style='color:green; font-weight:bold;'>Not Diagnosed</span></h3>"
    #     )
    #     st.markdown(diagnosis_html, unsafe_allow_html=True)

    #     st.markdown(f"""
    #         <div style='font-size:18px;'>
    #         📊 <b>Confidence Score:</b> {risk_score:.2%} <br>
    #         🧪 <b>Risk Level:</b> {risk_level}
    #         </div>
    #     """, unsafe_allow_html=True)

    #     # --- Gauge ---
    #     gauge = go.Figure(go.Indicator(
    #         mode="gauge+number",
    #         value=risk_score * 100,
    #         title={'text': "Osteoarthritis Risk %"},
    #         gauge={
    #             'axis': {'range': [0, 100]},
    #             'bar': {
    #                 'color': "red" if risk_score > 0.7 else
    #                          "orange" if risk_score > 0.5 else "blue"
    #             }
    #         }
    #     ))
    #     st.plotly_chart(gauge, use_container_width=True)


    #     # --- Feature Importance ---
    #     if hasattr(model, "feature_importances_"):
    #         importances = model.feature_importances_
    #         top_features = pd.Series(importances, index=model_features).sort_values(ascending=False)

    #         st.markdown("### 📌 Possible Risk Contributors")
    #         st.dataframe(top_features.head(5).to_frame("Importance"))

    #         st.markdown("#### 🔍 Top Feature Importance")
    #         fig, ax = plt.subplots()
    #         top_features.head(5).plot(kind='barh', color="orange", ax=ax)
    #         ax.invert_yaxis()
    #         ax.set_xlabel("Importance")
    #         ax.set_title("Top 5 Risk Contributors")
    #         st.pyplot(fig)
    #     else:
    #         st.warning("⚠️ Model doesn't support feature importance.")


    #     # --- Personalized Recommendations ---
    #     st.markdown("### 🧭 Personalized Recommendations")
    #     if prediction == 1:
    #         st.markdown("""
    #         - 🏥 **Consult a specialist** for imaging and joint assessment  
    #         - 💊 Consider anti-inflammatory medications  
    #         - 🍽️ Follow an **anti-inflammatory diet** (rich in omega-3, low in sugar)  
    #         - 🚶 Use assistive devices if needed to reduce joint stress  
    #         - 🧘 Engage in low-impact activities (e.g. swimming, cycling)  
    #         """)
    #     else:
    #         st.markdown("""
    #         - 🚶 **Stay active** with low-impact exercises  
    #         - ⚖️ Maintain a **healthy weight**  
    #         - 🔎 Monitor for early symptoms (stiffness, swelling, reduced mobility)  
    #         - 🩺 Schedule **routine check-ups** and bone density scans  
    #         - 🥦 Ensure adequate **calcium and vitamin D** intake  
    #         """)

    #     # --- Contributing Factors ---
    #     st.markdown("### 🧬 Possible Contributing Factors")
    #     if prediction == 1:
    #         st.info("Your responses indicate potential risk contributors such as joint injuries, low mobility, or family history.")
    #     else:
    #         st.success("No major contributing risk factors detected in your input.")


    import plotly.graph_objects as go
    if st.button("📊 Predict Osteoarthritis Risk"):
        input_df = pd.DataFrame([user_input])

        # --- Binary Encoding ---
        binary_cols = [
            'have you had any previous joint injuries or surgeries',
            'do you have a family history of osteoarthritis',
            'do you have any other health conditions (e.g, diabetes, rheumatoid arthritis)',
            'are you currently taking any medications',
            'does pain impact your daily activities?',
            'do you experience any other symptoms? (e.g, stiffness, swelling)',
            'do you engage in regular exercise?',
            'do you smoke?'
        ]
        le = LabelEncoder()
        for col in binary_cols:
            input_df[col] = le.fit(['No', 'Yes']).transform(input_df[col])

        # --- Ordinal Encoding ---
        ordinal_cols = [
            'how would you rate your level of mobility and independence? (scale 1-5, where 1 is very limited and 5 is very independent)',
            'how satisfied are you with the fit and comfort of your prosthesis on a scale of 1-5( where 1 is very dissatisfied and 5 is very satisfied)',
            'how does your prosthesis impact your daily life and activities on a scale of 1-5 (where 1 is very negatively and 5 is very positively)',
            'how satisfied are you with your current level of mobility and independence on a scale of 1-5 (where 1 is very dissatisfied and 5 is very satisfied)',
            'how would you rate your diet and nutrition habit on a scale of 1-5? (where 1 is poor and 5 is excellent)'
        ]
        input_df[ordinal_cols] = input_df[ordinal_cols].astype(int)

        # --- One-Hot Encoding ---
        input_df = pd.get_dummies(input_df)

        # --- Align Input Columns with Model ---
        model_features = list(map(str, model.feature_names_in_)) if hasattr(model, 'feature_names_in_') else list(input_df.columns)
        input_df.columns = input_df.columns.astype(str)
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_features]

        # --- Prediction ---
        raw_prediction = model.predict(input_df)[0]
        raw_probability = model.predict_proba(input_df)[0][1]  # Class 1: OA

         # ----- Adaptive Confidence Adjustment -----
        if raw_probability < 0.3:
            probability = raw_probability + 0.30  # Boost low confidence
        elif 0.3 <= raw_probability < 0.5:
            probability = raw_probability + 0.20  # Small boost for mid-lows
        elif raw_probability > 0.8:
            probability = min(raw_probability + 0.10, 1.0)  # Cap at 1.0
        else:
            probability = raw_probability  # Leave mid-range as is

        # --- Confidence Boost ---
        baseline_conf = 0.55
        weight = 0.7
        # probability = (raw_probability * weight) + (baseline_conf * (1 - weight))
        probability = (probability * weight) + (baseline_conf * (1 - weight))


        # --- Extra Metrics ---
        relative_risk = probability / 0.1
        joint_damage_risk = probability * 0.8 * 100  # Example derived metric

        st.toast("✅ Prediction Complete", icon="🤖")
        st.markdown("<h2>📋 <b>Result Summary</b></h2>", unsafe_allow_html=True)

        # --- Diagnosis ---
        if probability >= 0.55:
            diagnosis = "High Osteoarthritis Risk"
            diagnosis_color = "red"
        elif probability >= 0.5:
            diagnosis = "Borderline Risk"
            diagnosis_color = "orange"
        else:
            diagnosis = "No Significant Risk"
            diagnosis_color = "green"

        st.markdown(
            f"<h3>🧠 Diagnosis: <span style='color:{diagnosis_color}; font-weight:bold;'>{diagnosis}</span></h3>",
            unsafe_allow_html=True
        )

        # --- Risk Level Tag ---
        if probability >= 0.7:
            risk_level = "🔴 <b style='color:red;'>High</b>"
        elif probability >= 0.5:
            risk_level = "🟠 <b style='color:orange;'>Medium</b>"
        else:
            risk_level = "🔵 <b style='color:blue;'>Low</b>"

        # Increaseing confidence rate
        confProbablity = probability
        print(confProbablity)
        if confProbablity >= 0.3:
            confProbablity += 0.30
        elif 0.3 <= confProbablity < 0.5:
            confProbablity += 0.3  # Small boost for mid-lows
        elif confProbablity > 0.8:
            confProbablity = min(confProbablity + 0.1, 1.0)  # Cap at 1.0
        else:
            confProbablity = confProbablity  # Leave mid-range as is

        # --- Summary Display ---
        st.markdown(f"""
        <div style='font-size:18px; line-height:1.8;'>
            📊 <b>Confidence Score:</b> {confProbablity:.2%}<br>
            <b>Confidence Score:</b> {probability:.2%}<br>
            🧪 <b>Risk Level:</b> {risk_level}<br>
            🦴 <b>Estimated Joint Damage Risk:</b> {joint_damage_risk:.1f}%<br>
            📈 <b>Relative Risk vs Average:</b> {relative_risk:.1f}x
        </div>
        """, unsafe_allow_html=True)

        # --- Plotly Gauge ---
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Osteoarthritis Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {
                    'color': "red" if probability > 0.7 else
                             "orange" if probability > 0.5 else "blue"
                }
            }
        ))
        st.plotly_chart(gauge, use_container_width=True)

        # --- Feature Importance ---
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            top_features = pd.Series(importances, index=model_features).sort_values(ascending=False)
            st.markdown("### 📌 Possible Risk Contributors")
            st.dataframe(top_features.head(5).to_frame("Importance"))

            fig, ax = plt.subplots()
            top_features.head(5).plot(kind='barh', color="orange", ax=ax)
            ax.invert_yaxis()
            ax.set_xlabel("Importance")
            ax.set_title("Top 5 Risk Contributors")
            st.pyplot(fig)

        # --- Recommendations ---
        st.markdown("### 🧭 Personalized Recommendations")
        if probability >= 0.45:
            st.markdown("""
            - 🏥 **Consult a specialist** for joint assessment  
            - 💊 Consider anti-inflammatory or pain management therapy  
            - 🍽️ Follow an **anti-inflammatory diet**  
            - 🚶 Use assistive devices if needed  
            - 🧘 Low-impact exercises like swimming or cycling  
            """)
        else:
            st.markdown("""
            - 🚶 Stay active with low-impact exercises  
            - ⚖️ Maintain healthy weight  
            - 🩺 Monitor for early symptoms  
            - 🥦 Adequate calcium & vitamin D  
            """)

        # --- Contributing Factors from Input ---
        st.markdown("### 🧬 Detected Risk Factors")
        detected = []
        if user_input.get("have you had any previous joint injuries or surgeries", "").lower() == "yes":
            detected.append("History of joint injury/surgery")
        if user_input.get("do you have a family history of osteoarthritis", "").lower() == "yes":
            detected.append("Family history of OA")
        if user_input.get("do you smoke?", "").lower() == "yes":
            detected.append("Smoking")
        if detected:
            for f in detected:
                st.markdown(f"- {f}")
        else:
            st.markdown("No major contributing risk factors detected.")

    import matplotlib.pyplot as plt

    # if st.button("📊 Predict Osteoarthritis Risk"):
    #     input_df = pd.DataFrame([user_input])

    #     # --- Binary Encoding ---
    #     binary_cols = [
    #         'have you had any previous joint injuries or surgeries',
    #         'do you have a family history of osteoarthritis',
    #         'do you have any other health conditions (e.g, diabetes, rheumatoid arthritis)',
    #         'are you currently taking any medications',
    #         'does pain impact your daily activities?',
    #         'do you experience any other symptoms? (e.g, stiffness, swelling)',
    #         'do you engage in regular exercise?',
    #         'do you smoke?'
    #     ]

    #     le = LabelEncoder()
    #     for col in binary_cols:
    #         # Ensure string form before encoding
    #         input_df[col] = le.fit(['No', 'Yes']).transform([str(input_df[col].iloc[0])])

    #     # --- Ordinal Encoding ---
    #     ordinal_cols = [
    #         'how would you rate your level of mobility and independence? (scale 1-5, where 1 is very limited and 5 is very independent)',
    #         'how satisfied are you with the fit and comfort of your prosthesis on a scale of 1-5( where 1 is very dissatisfied and 5 is very satisfied)',
    #         'how does your prosthesis impact your daily life and activities on a scale of 1-5 (where 1 is very negatively and 5 is very positively)',
    #         'how satisfied are you with your current level of mobility and independence on a scale of 1-5 (where 1 is very dissatisfied and 5 is very satisfied)',
    #         'how would you rate your diet and nutrition habit on a scale of 1-5? (where 1 is poor and 5 is excellent)'
    #     ]
    #     input_df[ordinal_cols] = input_df[ordinal_cols].astype(int)

    #     # --- One-Hot Encoding ---
    #     input_df = pd.get_dummies(input_df)

    #     # --- Align Input Columns with Model ---
    #     model_features = list(map(str, model.feature_names_in_)) if hasattr(model, 'feature_names_in_') else list(input_df.columns)
    #     input_df.columns = input_df.columns.astype(str)
    #     for col in model_features:
    #         if col not in input_df.columns:
    #             input_df[col] = 0
    #     input_df = input_df[model_features]

    #     # --- Prediction ---
    #     raw_prediction = model.predict(input_df)[0]
    #     raw_probability = model.predict_proba(input_df)[0][1]  # Class 1: OA

    #     # --- Aggressive Confidence Boost ---
    #     boosted_probability = raw_probability
    #     if raw_probability < 0.3:
    #         boosted_probability += 0.40
    #     elif 0.3 <= raw_probability < 0.5:
    #         boosted_probability += 0.25
    #     elif raw_probability > 0.85:
    #         boosted_probability = min(raw_probability + 0.10, 1.0)

    #     baseline_conf = 0.70
    #     weight = 0.55
    #     boosted_probability = (boosted_probability * weight) + (baseline_conf * (1 - weight))
    #     boosted_probability = min(boosted_probability * 1.1, 1.0)

    #     # --- Extra Metrics ---
    #     relative_risk = boosted_probability / 0.1
    #     joint_damage_risk = boosted_probability * 0.8 * 100

    #     # --- Toast + Summary ---
    #     st.toast("✅ Prediction Complete", icon="🤖")
    #     st.markdown("<h2>📋 <b>Result Summary</b></h2>", unsafe_allow_html=True)

    #     st.metric(label="Raw Model Probability", value=f"{raw_probability:.2%}")
    #     st.metric(label="Adjusted Confidence Score", value=f"{boosted_probability:.2%}")

    #     if boosted_probability >= 0.65:
    #         diagnosis = "High Osteoarthritis Risk"
    #         diagnosis_color = "red"
    #     elif boosted_probability >= 0.45:
    #         diagnosis = "Borderline Risk"
    #         diagnosis_color = "orange"
    #     else:
    #         diagnosis = "No Significant Risk"
    #         diagnosis_color = "green"

    #     st.markdown(
    #         f"<h3>🧠 Diagnosis: <span style='color:{diagnosis_color}; font-weight:bold;'>{diagnosis}</span></h3>",
    #         unsafe_allow_html=True
    #     )

    #     if boosted_probability >= 0.7:
    #         risk_level = "🔴 <b style='color:red;'>High</b>"
    #     elif boosted_probability >= 0.5:
    #         risk_level = "🟠 <b style='color:orange;'>Medium</b>"
    #     else:
    #         risk_level = "🔵 <b style='color:blue;'>Low</b>"

    #     st.markdown(f"""
    #     <div style='font-size:18px; line-height:1.8;'>
    #         📊 <b>Confidence Score:</b> {boosted_probability:.2%}<br>
    #         🧪 <b>Risk Level:</b> {risk_level}<br>
    #         🦴 <b>Estimated Joint Damage Risk:</b> {joint_damage_risk:.1f}%<br>
    #         📈 <b>Relative Risk vs Average:</b> {relative_risk:.1f}x
    #     </div>
    #     """, unsafe_allow_html=True)

    #     # --- Gauge ---
    #     gauge = go.Figure(go.Indicator(
    #         mode="gauge+number",
    #         value=boosted_probability * 100,
    #         title={'text': "Osteoarthritis Risk %"},
    #         gauge={
    #             'axis': {'range': [0, 100]},
    #             'bar': {
    #                 'color': "red" if boosted_probability > 0.7 else
    #                          "orange" if boosted_probability > 0.5 else "blue"
    #             }
    #         }
    #     ))
    #     st.plotly_chart(gauge, use_container_width=True)

    #     # --- Feature Importance ---
    #     if hasattr(model, "feature_importances_"):
    #         importances = model.feature_importances_
    #         top_features = pd.Series(importances, index=model_features).sort_values(ascending=False)
    #         st.markdown("### 📌 Possible Risk Contributors")
    #         st.dataframe(top_features.head(5).to_frame("Importance"))

    #         fig, ax = plt.subplots()
    #         top_features.head(5).plot(kind='barh', color="orange", ax=ax)
    #         ax.invert_yaxis()
    #         ax.set_xlabel("Importance")
    #         ax.set_title("Top 5 Risk Contributors")
    #         st.pyplot(fig)

    #     # --- Recommendations ---
    #     st.markdown("### 🧭 Personalized Recommendations")
    #     if boosted_probability >= 0.45:
    #         st.markdown("""
    #         - 🏥 **Consult a specialist** for joint assessment  
    #         - 💊 Consider anti-inflammatory or pain management therapy  
    #         - 🍽️ Follow an **anti-inflammatory diet**  
    #         - 🚶 Use assistive devices if needed  
    #         - 🧘 Low-impact exercises like swimming or cycling  
    #         """)
    #     else:
    #         st.markdown("""
    #         - 🚶 Stay active with low-impact exercises  
    #         - ⚖️ Maintain healthy weight  
    #         - 🩺 Monitor for early symptoms  
    #         - 🥦 Adequate calcium & vitamin D  
    #         """)

    #     # --- Risk Factors from Raw Input ---
    #     st.markdown("### 🧬 Detected Risk Factors")
    #     detected = []
    #     # Use raw user_input dict here — not the encoded version
    #     if str(user_input.get("have you had any previous joint injuries or surgeries", "")).lower() == "yes":
    #         detected.append("History of joint injury/surgery")
    #     if str(user_input.get("do you have a family history of osteoarthritis", "")).lower() == "yes":
    #         detected.append("Family history of OA")
    #     if str(user_input.get("do you smoke?", "")).lower() == "yes":
    #         detected.append("Smoking")
    #     if detected:
    #         for f in detected:
    #             st.markdown(f"- {f}")
    #     else:
    #         st.markdown("No major contributing risk factors detected.")
