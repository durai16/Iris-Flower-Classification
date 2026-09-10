from pathlib import Path
import joblib


# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "iris_model.pkl"


FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def predict_new_iris(data: dict):
    """
    Validate a new Iris input and return prediction details.
    """

    # 1. Check input type
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary.")

    # 2. Check required fields
    missing_fields = [field for field in FEATURES if field not in data]

    if missing_fields:
        raise ValueError(
            f"Missing required fields: {', '.join(missing_fields)}"
        )

    # 3. Check data types and values
    values = []

    for field in FEATURES:
        value = data[field]

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{field} must be a numeric value."
            )

        if value <= 0:
            raise ValueError(
                f"{field} must be greater than 0."
            )

        values.append(float(value))

    # 4. Load model
    model = joblib.load(MODEL_PATH)

    # 5. Make prediction
    prediction = model.predict([values])[0]
    probabilities = model.predict_proba([values])[0]

    class_names = [
        "Iris Setosa",
        "Iris Versicolor",
        "Iris Virginica",
    ]

    predicted_class = class_names[prediction]
    confidence = float(probabilities[prediction])

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
    }


# --------------------------------------------------
# Valid input
# --------------------------------------------------

valid_input = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}

print("\n=== Valid Input ===")

try:
    result = predict_new_iris(valid_input)
    print(result)
except ValueError as error:
    print(f"Validation Error: {error}")


# --------------------------------------------------
# Test 1: Missing field
# --------------------------------------------------

missing_input = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
}

print("\n=== Missing Field Test ===")

try:
    result = predict_new_iris(missing_input)
    print(result)
except ValueError as error:
    print(f"Validation Error: {error}")


# --------------------------------------------------
# Test 2: Negative value
# --------------------------------------------------

negative_input = {
    "sepal_length": 5.1,
    "sepal_width": -3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}

print("\n=== Negative Value Test ===")

try:
    result = predict_new_iris(negative_input)
    print(result)
except ValueError as error:
    print(f"Validation Error: {error}")


# --------------------------------------------------
# Test 3: Wrong data type
# --------------------------------------------------

wrong_type_input = {
    "sepal_length": "five",
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}

print("\n=== Wrong Type Test ===")

try:
    result = predict_new_iris(wrong_type_input)
    print(result)
except ValueError as error:
    print(f"Validation Error: {error}")