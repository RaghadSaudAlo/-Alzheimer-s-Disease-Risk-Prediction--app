import streamlit as st
import pandas as pd
import pickle

# ---- Feature Ranges  ----
ranges = {
    "FunctionalAssessment": (0.000459594, 9.996467073),
    "ADL": (0.001287928, 9.999747122),
    "MemoryComplaints": (0.0, 1.0),
    "MMSE": (0,30),
    "BehavioralProblems": (0.0, 1.0),
    "SleepQuality": (0.00262866, 9.999840317),
    'Age' : (30, 100),
    "CholesterolHDL": (20.00343401, 100.98032408),
    "Hypertension": (0, 1),
    "FamilyHistoryAlzheimers": (0.0, 1.0),
    "CholesterolLDL": (50.0,200,0),
    "CardiovascularDisease": (0.0, 1.0),
    "Diabetes": (0.0, 1.0),
    "BMI": (10, 50),
    "Disorientation": (0, 1),
    "CholesterolTriglycerides": (50, 400)
}

# ---- Load model and scaler ----
@st.cache_resource
def load_model():
    return pickle.load(open('the_best_model.pkl', 'rb'))

@st.cache_resource
def load_scaler():
    return pickle.load(open('minmax_scaler.pkl', 'rb'))

best_model = load_model()
scaler = load_scaler()
#--------------------------------

def main():
    st.markdown("<h1 style='text-align: center; color: #ffffff; font-weight: bold;'>🧠 Alzheimer's Disease Risk Prediction </h1>", unsafe_allow_html=True)
    st.write("Please answer the following questions:")


    # ----  input widgets ----
    functional_assessment = st.slider("Can you perform daily tasks on your own?, 0 = cannot do at all, 10 = fully capable)", float(ranges["FunctionalAssessment"][0]), float(ranges["FunctionalAssessment"][1]), 5.0, 0.01)
    adl = st.slider("Do you rely on others in your daily life? (0 = dependent, 10 = independent)", float(ranges["ADL"][0]), float(ranges["ADL"][1]), 5.0, 0.01)
    memory_complaints = st.radio("Do you notice any memory problems?", [0.0, 1.0], format_func=lambda x: "Yes" if x == 1.0 else "No")
    mmse = st.slider("What is your memory test score? (MMSE Score, 0 = low, 30 = high)", float(ranges["MMSE"][0]), float(ranges["MMSE"][1]), 10.0, 0.01)
    behavioral_problems = st.radio("Have you noticed any behavioral problems?", [0.0, 1.0], format_func=lambda x: "Yes" if x == 1.0 else "No")
    sleep_quality = st.slider("Sleep Quality (0=Poor, 10=Excellent)", float(ranges["SleepQuality"][0]), float(ranges["SleepQuality"][1]), 7.0, 0.01)
    age = st.number_input("Age (years)", min_value=int(ranges["Age"][0]), max_value=int(ranges["Age"][1]), step=1)
    cholesterol_hdl = st.slider("What is your HDL (High-Density Lipoprotein) cholesterol level?", float(ranges["CholesterolHDL"][0]), float(ranges["CholesterolHDL"][1]), 50.0, 0.01)
    hypertension = st.radio("Have you been diagnosed with high blood pressure (Hypertension)?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    family_history = st.radio("Does anyone in your family have Alzheimer's disease? ", [0.0, 1.0], format_func=lambda x: "Yes" if x == 1.0 else "No")
    cholesterol_ldl = st.slider("What is your LDL (Low-Density Lipoprotein) cholesterol level? ", float(ranges["CholesterolLDL"][0]), float(ranges["CholesterolLDL"][1]), 100.0, 0.01)
    cardiovascular_disease = st.radio("Have you had heart or blood circulation problems?", [0.0, 1.0], format_func=lambda x: "Yes" if x == 1.0 else "No")
    diabetes = st.radio("Do you have diabetes?", [0.0, 1.0], format_func=lambda x: "Yes" if x == 1.0 else "No")
    bmi = st.slider("What is your Body Mass Index (BMI)?", float(ranges["BMI"][0]), float(ranges["BMI"][1]), 25.0, 0.01) # 25.0 default value if the user didint change the the radio 
    disorientation = st.radio("Do you sometimes feel lost or confused about your surroundings?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    cholesterol_triglycerides = st.slider("What is your blood triglyceride level? (mg/dL)", float(ranges["CholesterolTriglycerides"][0]), float(ranges["CholesterolTriglycerides"][1]), 150.0, 0.01)


    # ---- Collect input for prediction ----
    input_data = pd.DataFrame([{
        'FunctionalAssessment': functional_assessment,
        'ADL': adl,
        'MemoryComplaints': memory_complaints,
        'MMSE': mmse,
        'BehavioralProblems': behavioral_problems,
        'SleepQuality': sleep_quality,
        'Age': age,
        'CholesterolHDL': cholesterol_hdl,
        'Hypertension': hypertension,
        'FamilyHistoryAlzheimers': family_history,
        'CholesterolLDL': cholesterol_ldl,
        'CardiovascularDisease': cardiovascular_disease,
        'Diabetes': diabetes,
        'BMI': bmi,
        'Disorientation': disorientation,
        'CholesterolTriglycerides': cholesterol_triglycerides
    }])

    # ---- Prediction ----
    if st.button("Predict Alzheimer's Risk"):
        # Scale input using the same scaler as in training
        input_scaled = scaler.transform(input_data)
        prediction = best_model.predict(input_scaled)
        if prediction[0] == 1:
            st.error("You are at risk of having Alzheimer's disease.")
            st.warning("⚠️ Warning: Not a final diagnosis only prediction. This is not a medical diagnosis. Please consult a healthcare professional for further evaluation.")
        else:
            st.success("You are NOT likely to have Alzheimer's disease.")
            

if __name__ == '__main__':
    main()
