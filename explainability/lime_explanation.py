from pathlib import Path
import joblib
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_iris


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "iris_model.pkl"
OUTPUT_DIR = BASE_DIR / "explainability" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load Iris dataset
iris = load_iris()

X = iris.data
feature_names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

class_names = [
    "Iris Setosa",
    "Iris Versicolor",
    "Iris Virginica",
]


# Load the existing trained model
model = joblib.load(MODEL_PATH)


# LIME needs a prediction-probability function
def predict_proba(data):
    return model.predict_proba(data)


# Create LIME explainer
explainer = LimeTabularExplainer(
    training_data=X,
    feature_names=feature_names,
    class_names=class_names,
    mode="classification",
    discretize_continuous=True,
    random_state=42,
)


# Two individual samples
samples = {
    "setosa": X[0],
    "versicolor": X[50],
}


for sample_name, sample in samples.items():

    sample = np.array(sample)

    # Model prediction
    prediction = model.predict([sample])[0]
    probabilities = model.predict_proba([sample])[0]

    predicted_class = class_names[prediction]

    print("\n" + "=" * 60)
    print(f"LIME Explanation: {sample_name.upper()}")
    print("=" * 60)

    print(f"Input values: {sample}")
    print(f"Predicted class: {predicted_class}")

    print("\nPrediction probabilities:")

    for class_name, probability in zip(class_names, probabilities):
        print(f"{class_name}: {probability:.4f}")

    # Generate LIME explanation
    explanation = explainer.explain_instance(
        sample,
        predict_proba,
        num_features=4,
        top_labels=1,
    )

    # Print explanation
    print("\nFeature contributions:")

    for feature, weight in explanation.as_list(
        label=prediction
    ):
        print(f"{feature}: {weight:.4f}")

    # Save HTML explanation
    output_path = OUTPUT_DIR / f"lime_{sample_name}.html"

    explanation.save_to_file(str(output_path))

    print(f"\nLIME explanation saved to:")
    print(output_path)