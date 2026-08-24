from pathlib import Path
from uuid import uuid4
import joblib
import numpy as np
from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel,Field
import logging
from logging.handlers import RotatingFileHandler
import time

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "app.log"

# Set up logging with rotation
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=1_000_000, backupCount=3),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PredictionException(Exception):
    def __init__(self, message: str):
        self.message = message

# Load the trained model
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "iris_model.pkl"
model_loaded = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    global model_loaded
    model = joblib.load(MODEL_PATH)
    model_loaded = True
    logger.info("Iris model loaded successfully.")
    yield

# Create FastAPI application
app = FastAPI(
    title="Iris Flower Classification API",
    description="API for predicting Iris flower species using a Machine Learning model.",
    version="1.0.0",
    lifespan=lifespan
)

@app.exception_handler(PredictionException)
async def prediction_exception_handler(
    request: Request,
    exc: PredictionException
):
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "PredictionError",
            "message": exc.message
        }
    )
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id

    start_time = time.time()

    logger.info(
        f"Request started: {request.method} {request.url.path} "
        f"request_id={request_id}"
    )

    response = await call_next(request)

    duration = time.time() - start_time

    logger.info(
        f"Request completed: {request.method} {request.url.path} "
        f"status={response.status_code} "
        f"duration={duration:.4f}s "
        f"request_id={request_id}"
    )

    response.headers["X-Request-ID"] = request_id

    return response

# Define the input data structure
class IrisInput(BaseModel):
    sepal_length: float=Field(..., gt=0)
    sepal_width: float=Field(..., gt=0,le=10)
    petal_length: float=Field(..., gt=0)
    petal_width: float=Field(..., gt=0)
    
class PredictionOutput(BaseModel):
        prediction: str
        confidence: float
        request_id: str

# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Iris Flower Classification API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded
    }
    
# Prediction endpoint
@app.post("/predict",response_model=PredictionOutput)
def predict(data: IrisInput, request: Request):    
    request_id = request.state.request_id
    
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])
    try:
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
    except Exception as e:
        logger.error(
        f"Prediction failed: {e} request_id={request_id}"
    )
        raise PredictionException("Prediction failed")
    
    species = [
        "Iris Setosa",
        "Iris Versicolor",
        "Iris Virginica"
    ]
    predicted_species = species[prediction]
    confidence = float(probabilities[prediction])
    logger.info(
    f"Prediction successful: {predicted_species} "
    f"confidence={confidence:.2f} "
    f"request_id={request_id}"
)

    return {
        "prediction": predicted_species,
        "confidence": round(confidence, 2),
        "request_id": request_id
    }