from pathlib import Path
import joblib
from app.metrics import metrics
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import uuid
import logging
from logging.handlers import RotatingFileHandler
import time
from app.config import settings

# Paths

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "app.log"

MODEL_PATH = (settings.MODEL_PATH
)

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(
            LOG_FILE,
            maxBytes=1_000_000,
            backupCount=3
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Custom Exception

class PredictionException(Exception):

    def __init__(self, message: str):
        self.message = message

# Load Model

@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.model = joblib.load(MODEL_PATH)
    app.state.model_loaded = True

    logger.info("Iris model loaded successfully.")

    yield

# FastAPI Application

app = FastAPI(
title=settings.API_TITLE,    
description="API for predicting Iris flower species using a Machine Learning model.",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.CORS_ORIGINS.split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# Exception Handler

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


# Logging Middleware

@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.time()
    logger.info(
        f"Request started: {request.method} "
        f"{request.url.path} "
        f"request_id={request_id}"
    )

    try:
        response = await call_next(request)

        duration = time.time() - start_time

        metrics.record_request(
            endpoint=request.url.path,
            status_code=response.status_code,
            duration=duration
        )

        logger.info(
            f"Request completed: {request.method} "
            f"{request.url.path} "
            f"status={response.status_code} "
            f"duration={duration:.3f}s "
            f"request_id={request_id}"
        )

        return response

    except Exception:
        duration = time.time() - start_time

        metrics.record_request(
            endpoint=request.url.path,
            status_code=500,
            duration=duration
        )

        raise
@app.get("/metrics")
async def get_metrics():
    return metrics.get_metrics()
# Home Endpoint

@app.get("/")
def home():

    return {
        "message": "Iris Flower Classification API is running"
    }


# Include API v1 Router

from app.routers.v1 import router as v1_router
from app.routers.v2 import router as v2_router
app.include_router(v1_router)
app.include_router(v2_router)