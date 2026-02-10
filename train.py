import os
import pickle
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# ---------- Load your dataset ----------
# Put the CSV in your project root (e.g., Perfomance-prediction-of-students-Using-ML/data.csv)
df = pd.read_csv("data.csv")  # change if your filename is different

# ---------- Separate features and target ----------
X = df.drop(columns=["math_score"])  # your target column
y = df["math_score"]

# ---------- Define preprocessing pipelines ----------
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
    ],
    verbose_feature_names_out=False
)

# ---------- Fit preprocessor and transform ----------
X_processed = preprocessor.fit_transform(X)

# ---------- Train model ----------
model = LinearRegression()
model.fit(X_processed, y)

# ---------- Save artifacts ----------
os.makedirs("artifacts", exist_ok=True)

with open("artifacts/preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Training completed. Artifacts saved!")
