import numpy as np

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from fastapi import Depends
from app.security import verify_api_key
from app.main import PredictionException, logger


# ==================================================
# API V2 Router
# ==================================================

router = APIRouter(
    prefix="/api/v2",
    tags=["API v2"],
    dependencies=[Depends(verify_api_key)]
)


# ==================================================
# Input Schema
# ==================================================

class IrisInputV2(BaseModel):

    sepal_length: float = Field(..., gt=0)

    sepal_width: float = Field(
        ...,
        gt=0,
        le=10
    )

    petal_length: float = Field(..., gt=0)

    petal_width: float = Field(..., gt=0)


# ==================================================
# V2 Output Schema
# ==================================================

class PredictionOutputV2(BaseModel):

    prediction: str

    probabilities: dict[str, float]

    model_version: str

    request_id: str


# ==================================================
# V2 Prediction Endpoint
# ==================================================

@router.post(
    "/predict",
    response_model=PredictionOutputV2
)
def predict_v2(
    data: IrisInputV2,
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
            f"V2 prediction failed: {e} "
            f"request_id={request_id}"
        )

        raise PredictionException(
            "V2 prediction failed"
        )

    species = [
        "Iris Setosa",
        "Iris Versicolor",
        "Iris Virginica"
    ]

    prediction = int(prediction)

    predicted_species = species[prediction]

    probability_distribution = {
        species[index]: round(float(probability), 4)
        for index, probability in enumerate(probabilities)
    }

    logger.info(
        f"V2 prediction successful: "
        f"{predicted_species} "
        f"request_id={request_id}"
    )

    return PredictionOutputV2(
        prediction=predicted_species,
        probabilities=probability_distribution,
        model_version="2.0.0",
        request_id=request_id
    )