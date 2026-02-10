import streamlit as st
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# App title
st.set_page_config(page_title="Student Performance Prediction")
st.title("🎓 Student Performance Prediction")

st.write("Enter student details to predict performance")

# ---- Input Fields ----
gender = st.selectbox("Gender", ["male", "female"])

ethnicity = st.selectbox(
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

lunch = st.selectbox("Lunch", ["standard", "free/reduced"])

test_preparation_course = st.selectbox(
    "Test Preparation Course",
    ["none", "completed"]
)

reading_score = st.number_input(
    "Reading Score", min_value=0.0, max_value=100.0, step=1.0
)

writing_score = st.number_input(
    "Writing Score", min_value=0.0, max_value=100.0, step=1.0
)

# ---- Prediction ----
if st.button("Predict Performance"):
    try:
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        st.success(f"✅ Predicted Math Score: {results[0]}")

    except Exception as e:
        st.error("Something went wrong during prediction")
        st.write(e)
