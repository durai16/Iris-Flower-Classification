from pathlib import Path
import joblib


BASE_DIR = Path(__file__).resolve().parent.parent
PIPELINE_PATH = BASE_DIR / "models" / "iris_pipeline.pkl"


class_names = [
    "Iris Setosa",
    "Iris Versicolor",
    "Iris Virginica",
]


def predict_iris(data):
    """Load the saved pipeline and predict a new Iris sample."""

    pipeline = joblib.load(PIPELINE_PATH)

    prediction = pipeline.predict([data])[0]
    probabilities = pipeline.predict_proba([data])[0]

    return {
        "prediction": class_names[prediction],
        "confidence": round(float(probabilities[prediction]), 4),
    }


if __name__ == "__main__":

    sample = [5.1, 3.5, 1.4, 0.2]

    result = predict_iris(sample)

    print("\n=== Reusable Pipeline Prediction ===")
    print(f"Input: {sample}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")