# Iris Flower Classification API

*A simple Machine Learning REST API for predicting Iris flower species from flower measurements.*

## Overview

This project demonstrates how to build and expose a Machine Learning classification model through a REST API. The project uses the well-known Iris dataset provided by scikit-learn and a Logistic Regression model to predict the species of an Iris flower.

The main purpose of this project is to understand the engineering process of converting a Machine Learning model into a usable API service.

## Objective

The objective is to build a REST API that accepts four measurements of an Iris flower and predicts its species.

The API will accept:

* Sepal length
* Sepal width
* Petal length
* Petal width

The API will return:

* Predicted Iris species
* Prediction confidence

## Dataset

**Dataset:** Iris Dataset
**Source:** scikit-learn built-in dataset

The dataset contains measurements of three Iris flower species:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

### Input Features

```text
sepal_length
sepal_width
petal_length
petal_width
```

### Target

```text
Iris species
```

## Machine Learning Problem

This project solves a **supervised classification problem**.

The model learns patterns from the four flower measurements and predicts one of the three Iris species.

**Selected Model:** Logistic Regression

Logistic Regression is selected because it is simple, fast, and suitable for this classification problem.

## API Contract

The `/predict` endpoint will accept a `POST` request containing the four Iris flower measurements.

### Request

```json
{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}
```

### Response

```json
{
    "prediction": "Iris Setosa",
    "confidence": 0.98
}
```

If the input is missing, invalid, or incorrectly formatted, the API will return an appropriate error response.

## API Endpoint

| Method | Endpoint   | Description                     |
| ------ | ---------- | ------------------------------- |
| POST   | `/predict` | Predict the Iris flower species |

## Request Flow

```text
Client
  |
  v
POST /predict
  |
  v
Input Validation
  |
  v
Feature Preparation
  |
  v
Trained ML Model
  |
  v
Prediction
  |
  v
JSON Response
```

### Flow Explanation

1. The client sends Iris flower measurements to the `/predict` endpoint.
2. The API receives and validates the input.
3. The validated features are prepared for the Machine Learning model.
4. The trained model processes the input.
5. The model predicts the Iris flower species.
6. The API returns the prediction and confidence score as a JSON response.

## Model vs Service

### Machine Learning Model

The Machine Learning model learns patterns from the Iris dataset and makes predictions for new input data.

```text
Input Features
      |
      v
ML Model
      |
      v
Prediction
```

### API Service

The API service provides an interface that allows external applications to communicate with the Machine Learning model.

```text
Client
  |
  v
API Service
  |
  v
ML Model
  |
  v
Response
```

In simple terms:

```text
Model   = Makes the prediction
Service = Makes the model accessible
```

## REST API Concepts

This project will use basic REST API concepts.

### HTTP Method

`POST` will be used for `/predict` because the client sends flower measurements to the API.

### HTTP Status Codes

The API will use standard HTTP status codes:

```text
200 OK
```

Prediction completed successfully.

```text
400 Bad Request
```

The input data is invalid or incomplete.

```text
500 Internal Server Error
```

An unexpected server-side error occurred.

## Minimum Viable Product

The first version of the project will focus on the core Machine Learning API functionality.

### MVP Features

* Load the Iris dataset
* Prepare the dataset
* Train a classification model
* Save the trained model
* Create a FastAPI application
* Create the `/predict` endpoint
* Validate input data
* Generate predictions
* Return prediction results as JSON

### Out of Scope for the MVP

* User authentication
* Database integration
* Frontend application
* Admin dashboard
* Complex Machine Learning algorithms

## Technology Stack

```text
Programming Language : Python
Machine Learning     : Scikit-learn
API Framework        : FastAPI
Server               : Uvicorn
Data Processing      : NumPy / Pandas
Version Control      : Git
Repository           : GitHub
```

## Planned Project Structure

The project structure will be implemented in the next task.

```text
iris-ml-api/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── model.py
│
├── models/
│   └── iris_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Project Architecture

```text
                 Iris ML API
                      |
                      v
                FastAPI Server
                      |
                      v
                 POST /predict
                      |
                      v
                Input Validation
                      |
                      v
                Feature Preparation
                      |
                      v
             Logistic Regression
                      |
                      v
                  Prediction
                      |
                      v
                JSON Response
```

## Task 1 Scope

Task 1 focuses on understanding the project and planning the architecture before writing the application code.

### Task 1 Deliverables

* Dataset selected
* ML problem selected
* ML model selected
* API endpoint defined
* API input and output defined
* API contract documented
* Request-to-response flow designed
* Technology stack selected
* Project structure planned
* GitHub repository created

## Future Development

The project will be developed in the following stages:

```text
Task 1 → Understand the project and plan the architecture
Task 2 → Create project structure and Python environment
Task 3 → Prepare dataset and train the ML model
Task 4 → Build the FastAPI service
Task 5 → Implement validation and prediction
Task 6 → Test the API
Task 7 → Document and deploy the application
```

## Expected Final Result

The final application will allow a client to send Iris flower measurements to the `/predict` endpoint and receive the predicted flower species.

```text
Client
  |
  | Flower Measurements
  v
FastAPI
  |
  | Input Validation
  v
ML Model
  |
  | Prediction
  v
JSON Response
```

Example response:

```json
{
    "prediction": "Iris Setosa",
    "confidence": 0.98
}
```

## Conclusion

The Iris Flower Classification API is a beginner-friendly Machine Learning project designed to demonstrate how a Machine Learning model can be integrated into a REST API service.

The project follows a planned development approach, starting with problem definition, API contract, and architecture before moving into implementation. This provides a clear foundation for developing a simple and maintainable Machine Learning API.
