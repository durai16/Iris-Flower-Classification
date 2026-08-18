from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target


# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)


# Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.2f}")


# Create the models directory
models_directory = Path(__file__).resolve().parent.parent / "models"
models_directory.mkdir(exist_ok=True)


# Save the trained model
model_path = models_directory / "iris_model.pkl"

joblib.dump(model, model_path)

print(f"Model saved to: {model_path}")
# Load the saved model
loaded_model = joblib.load(model_path)

# Make a sample prediction
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = loaded_model.predict(sample)
probability = loaded_model.predict_proba(sample)

predicted_species = iris.target_names[prediction[0]]
confidence = probability[0][prediction[0]]

print(f"Predicted species: {predicted_species}")
print(f"Confidence: {confidence:.2f}")