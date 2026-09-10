from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "iris_model.pkl"
OUTPUT_DIR = BASE_DIR / "explainability" / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load trained Logistic Regression model
model = joblib.load(MODEL_PATH)

# Iris feature names
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


# Logistic Regression coefficients
coefficients = model.coef_

# Calculate overall feature importance
# Mean absolute coefficient across all classes
feature_importance = np.mean(np.abs(coefficients), axis=0)

# Sort features from most to least important
indices = np.argsort(feature_importance)[::-1]

sorted_features = [feature_names[i] for i in indices]
sorted_importance = feature_importance[indices]


# Print results
print("\n=== Iris Model Feature Importance ===")

for feature, importance in zip(sorted_features, sorted_importance):
    print(f"{feature}: {importance:.4f}")


# Create chart
plt.figure(figsize=(9, 6))

plt.bar(sorted_features, sorted_importance)

plt.title("Iris Logistic Regression - Global Feature Importance")
plt.xlabel("Features")
plt.ylabel("Mean Absolute Coefficient")

plt.xticks(rotation=20)
plt.tight_layout()


# Save chart
output_path = OUTPUT_DIR / "feature_importance.png"

plt.savefig(output_path, dpi=300)

print(f"\nFeature importance chart saved to:")
print(output_path)

plt.show()