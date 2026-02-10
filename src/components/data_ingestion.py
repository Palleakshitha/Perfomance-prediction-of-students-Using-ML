import os
import pandas as pd
from src.exception import CustomException
import sys

class DataIngestion:
    def __init__(self):
        self.raw_data_path = os.path.join("artifacts", "data.csv")

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv("notebook/data/stud.csv")  # your dataset path
            os.makedirs("artifacts", exist_ok=True)
            df.to_csv(self.raw_data_path, index=False)
            return self.raw_data_path
        except Exception as e:
            raise CustomException(e, sys)
