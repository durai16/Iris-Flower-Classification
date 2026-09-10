from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_PATH = OUTPUT_DIR / "iris_pipeline.pkl"


# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# Create reusable pipeline
pipeline = Pipeline([
    (
        "model",
        LogisticRegression(max_iter=200)
    )
])


# Train pipeline
pipeline.fit(X_train, y_train)


# Evaluate
accuracy = pipeline.score(X_test, y_test)

print("\n=== Iris Pipeline ===")
print(f"Test Accuracy: {accuracy:.4f}")


# Save pipeline
joblib.dump(pipeline, PIPELINE_PATH)

print("\nPipeline saved to:")
print(PIPELINE_PATH)


# Test prediction
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = pipeline.predict(sample)[0]
probabilities = pipeline.predict_proba(sample)[0]

class_names = [
    "Iris Setosa",
    "Iris Versicolor",
    "Iris Virginica",
]

print("\n=== Sample Prediction ===")
print(f"Prediction: {class_names[prediction]}")
print(f"Confidence: {probabilities[prediction]:.4f}")