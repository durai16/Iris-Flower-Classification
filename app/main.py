from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel


# Create FastAPI application
app = FastAPI(
    title="Iris Flower Classification API",
    description="API for predicting Iris flower species using a Machine Learning model.",
    version="1.0.0"
)


# Load the trained model
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "iris_model.pkl"

model = joblib.load(MODEL_PATH)


# Define the input data structure
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Iris Flower Classification API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: IrisInput):

    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    species = [
        "Iris Setosa",
        "Iris Versicolor",
        "Iris Virginica"
    ]

    predicted_species = species[prediction]
    confidence = float(probabilities[prediction])

    return {
        "prediction": predicted_species,
        "confidence": round(confidence, 2)
    }