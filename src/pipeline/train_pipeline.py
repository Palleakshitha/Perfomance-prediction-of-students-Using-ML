import sys
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException

if __name__ == "__main__":
    try:
        ingestion = DataIngestion()
        data_path = ingestion.initiate_data_ingestion()

        df = pd.read_csv(data_path)

        transformation = DataTransformation()
        X, y = transformation.initiate_data_transformation(df)

        trainer = ModelTrainer()
        trainer.initiate_model_trainer(X, y)

        print("✅ Training completed successfully")

    except Exception as e:
        raise CustomException(e, sys)
