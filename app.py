import streamlit as st
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(page_title="Student Performance Prediction")
st.title("🎓 Student Performance Prediction")

st.write("Fill the student details to predict math score")

# ---------- INPUTS ----------
gender = st.selectbox("Gender", ["male", "female"])

race_ethnicity = st.selectbox(
    "Race / Ethnicity",
    ["group A", "group B", "group C", "group D", "group E"]
)

parental_level_of_education = st.selectbox(
    "Parental Level of Education",
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]
)

lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])

test_preparation_course = st.selectbox(
    "Test Preparation Course",
    ["none", "completed"]
)

reading_score = st.number_input(
    "Reading Score", min_value=0, max_value=100, value=50
)

writing_score = st.number_input(
    "Writing Score", min_value=0, max_value=100, value=50
)

# ---------- PREDICTION ----------
if st.button("Predict"):
    try:
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        final_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(final_df)

        st.success(f"✅ Predicted Math Score: {prediction[0]}")

    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)
