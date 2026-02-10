import os
import pickle
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# ---- LOAD DATA ----
df = pd.read_csv("data.csv")  # your dataset

X = df.drop(columns=["math_score"])
y = df["math_score"]

num_features = ["reading_score", "writing_score"]
cat_features = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course"
]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ("scaler", StandardScaler(with_mean=False))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

X_processed = preprocessor.fit_transform(X)

model = LinearRegression()
model.fit(X_processed, y)

os.makedirs("artifacts", exist_ok=True)

with open("artifacts/preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ NEW ARTIFACTS CREATED")
