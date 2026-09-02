from pathlib import Path
from uuid import uuid4
import joblib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

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
async def logging_middleware(request: Request, call_next):

    request_id = str(uuid4())

    request.state.request_id = request_id

    start_time = time.time()

    logger.info(
        f"Request started: "
        f"{request.method} "
        f"{request.url.path} "
        f"request_id={request_id}"
    )

    response = await call_next(request)

    duration = time.time() - start_time

    logger.info(
        f"Request completed: "
        f"{request.method} "
        f"{request.url.path} "
        f"status={response.status_code} "
        f"duration={duration:.4f}s "
        f"request_id={request_id}"
    )

    response.headers["X-Request-ID"] = request_id

    return response


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