import os
import sys
import pickle

from sklearn.linear_model import LinearRegression
from src.exception import CustomException

class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")

    def initiate_model_trainer(self, X, y):
        try:
            model = LinearRegression()
            model.fit(X, y)

            os.makedirs("artifacts", exist_ok=True)

            with open(self.model_path, "wb") as f:
                pickle.dump(model, f)

            return model

        except Exception as e:
            raise CustomException(e, sys)
