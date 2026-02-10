import os
import sys
import pickle
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.exception import CustomException

class DataTransformation:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def get_data_transformer_object(self):
        try:
            num_features = ["reading_score", "writing_score"]
            cat_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, num_features),
                    ("cat_pipeline", cat_pipeline, cat_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, df):
        try:
            preprocessor = self.get_data_transformer_object()
            X = df.drop(columns=["math_score"], axis=1)
            y = df["math_score"]

            X_processed = preprocessor.fit_transform(X)

            os.makedirs("artifacts", exist_ok=True)

            with open(self.preprocessor_path, "wb") as f:
                pickle.dump(preprocessor, f)

            return X_processed, y

        except Exception as e:
            raise CustomException(e, sys)
