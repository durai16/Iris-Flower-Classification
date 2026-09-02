import numpy as np
import time

from typing import List

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.main import PredictionException, logger
from app.config import settings


# ==================================================
# API V1 Router
# ==================================================

router = APIRouter(
    prefix="/api/v1",
    tags=["API v1"]
)
# API Versioning Plan:
# V1 keeps the original response contract for existing clients.
# If a future V2 requires a breaking response change, it will use
# a separate /api/v2 router and schema instead of modifying V1.
# For example, V2 can return the full probability distribution
# and model_version while V1 continues returning confidence.


# ==================================================
# Input Schema
# ==================================================

class IrisInput(BaseModel):

    sepal_length: float = Field(..., gt=0)

    sepal_width: float = Field(
        ...,
        gt=0,
        le=10
    )

    petal_length: float = Field(..., gt=0)

    petal_width: float = Field(..., gt=0)


# ==================================================
# Output Schema
# ==================================================

class PredictionOutput(BaseModel):

    prediction: str

    confidence: float

    request_id: str


# ==================================================
# Batch Input Schema
# ==================================================

class PredictionBatchInput(BaseModel):

    inputs: List[IrisInput]


# ==================================================
# Batch Output Schema
# ==================================================

class PredictionBatchOutput(BaseModel):

    predictions: List[PredictionOutput]


# ==================================================
# Health Endpoint
# ==================================================

@router.get("/health")
def health(request: Request):

    return {
        "status": "ok",
        "model_loaded": request.app.state.model_loaded
    }


# ==================================================
# Prediction Endpoint
# ==================================================

@router.post(
    "/predict",
    response_model=PredictionOutput
)
def predict(
    data: IrisInput,
    request: Request
):

    request_id = request.state.request_id

    model = request.app.state.model

    features = np.array([
        [
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]
    ])

    try:

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)[0]

    except Exception as e:

        logger.error(
            f"Prediction failed: {e} "
            f"request_id={request_id}"
        )

        raise PredictionException(
            "Prediction failed"
        )

    species = [
        "Iris Setosa",
        "Iris Versicolor",
        "Iris Virginica"
    ]

    predicted_species = species[prediction]

    confidence = float(
        probabilities[prediction]
    )

    logger.info(
        f"Prediction successful: "
        f"{predicted_species} "
        f"confidence={confidence:.2f} "
        f"request_id={request_id}"
    )

    return {
        "prediction": predicted_species,
        "confidence": round(confidence, 2),
        "request_id": request_id
    }


# ==================================================
# Batch Prediction Endpoint
# ==================================================

@router.post(
    "/predict-batch",
    response_model=PredictionBatchOutput
)
def predict_batch(
    data: PredictionBatchInput,
    request: Request
):

    start_time = time.time()

    request_id = request.state.request_id

    model = request.app.state.model

    batch_size = len(data.inputs)

    logger.info(
        f"Batch prediction started: "
        f"batch_size={batch_size} "
        f"request_id={request_id}"
    )

    # Validate batch size

    if batch_size == 0:

        raise PredictionException(
            "Batch input cannot be empty"
        )

    if batch_size > settings.MAX_BATCH_SIZE:
        raise PredictionException(
        f"Batch size cannot exceed {settings.MAX_BATCH_SIZE}"
    )

    # Create feature array

    features = np.array([
        [
            item.sepal_length,
            item.sepal_width,
            item.petal_length,
            item.petal_width
        ]
        for item in data.inputs
    ])

    try:

        # Predict entire batch at once

        predictions = model.predict(features)

        probabilities = model.predict_proba(features)

    except Exception as e:

        logger.error(
            f"Batch prediction failed: {e} "
            f"request_id={request_id}"
        )

        raise PredictionException(
            "Batch prediction failed"
        )

    species = [
        "Iris Setosa",
        "Iris Versicolor",
        "Iris Virginica"
    ]

    results = []

    for prediction, probability in zip(
        predictions,
        probabilities
    ):

        prediction = int(prediction)

        predicted_species = species[prediction]

        confidence = float(
            probability[prediction]
        )

        results.append(
            PredictionOutput(
                prediction=predicted_species,
                confidence=round(confidence, 2),
                request_id=request_id
            )
        )

    duration = time.time() - start_time

    logger.info(
        f"Batch prediction completed: "
        f"batch_size={batch_size} "
        f"duration={duration:.4f}s "
        f"request_id={request_id}"
    )

    return PredictionBatchOutput(
        predictions=results
    )


# ==================================================
# Model Information Endpoint
# ==================================================

@router.get("/model-info")
def model_info(request: Request):

    model = request.app.state.model

    return {
        "model_type": type(model).__name__,
        "model_version": "1.0.0",
        "training_date": "2026",
        "expected_features": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    }

