"""
FastAPI service for PCam center-region detection.

SECURITY FEATURES:
- Optional API key authentication (environment-based)
- Rate limiting per API key
- Request logging and audit trail
- Metrics and monitoring (Prometheus-compatible)
"""

import io
import os
import time
import uuid
import threading
import yaml
import logging
from typing import List, Optional, Dict
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Depends, Header, status
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from PIL import Image

from src.inference.predictor import PCamPredictor
from src.inference.ensemble_predictor import EnsemblePredictor
from src.utils.logging_utils import setup_logger


logger = setup_logger(json_format=True)
app = FastAPI(title="PCam Center-Region Detection API", version="1.0.0")

# Configuration from environment
API_KEY_ENABLED = os.getenv("API_KEY_ENABLED", "false").lower() == "true"
API_KEY = os.getenv("API_KEY", None)
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))


class RateLimiter:
    """
    Token-bucket rate limiter per API key.
    """
    def __init__(self, requests_per_window: int, window_sec: int):
        self.lock = threading.Lock()
        self.requests_per_window = requests_per_window
        self.window_sec = window_sec
        self.buckets: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        if not RATE_LIMIT_ENABLED:
            return True
        
        with self.lock:
            now = time.time()
            cutoff = now - self.window_sec
            
            # Remove old requests
            self.buckets[client_id] = [ts for ts in self.buckets[client_id] if ts > cutoff]
            
            # Check limit
            if len(self.buckets[client_id]) >= self.requests_per_window:
                return False
            
            # Record request
            self.buckets[client_id].append(now)
            return True
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        with self.lock:
            now = time.time()
            cutoff = now - self.window_sec
            valid_requests = [ts for ts in self.buckets[client_id] if ts > cutoff]
            return max(0, self.requests_per_window - len(valid_requests))


class MetricsStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_requests = 0
        self.error_requests = 0
        self.predictions = 0
        self.latency_ms_sum = 0.0
        self.latency_ms_count = 0
        self.auth_failures = 0
        self.rate_limit_violations = 0

    def record(
        self, 
        latency_ms: float, 
        predictions: int, 
        error: bool,
        auth_failure: bool = False,
        rate_limited: bool = False,
    ):
        with self.lock:
            self.total_requests += 1
            self.error_requests += 1 if error else 0
            self.predictions += predictions
            self.latency_ms_sum += latency_ms
            self.latency_ms_count += 1
            self.auth_failures += 1 if auth_failure else 0
            self.rate_limit_violations += 1 if rate_limited else 0

    def render_prometheus(self) -> str:
        avg_latency = (
            self.latency_ms_sum / self.latency_ms_count
            if self.latency_ms_count > 0
            else 0.0
        )
        return "\n".join(
            [
                "# TYPE pcam_requests_total counter",
                f"pcam_requests_total {self.total_requests}",
                "# TYPE pcam_requests_errors_total counter",
                f"pcam_requests_errors_total {self.error_requests}",
                "# TYPE pcam_requests_auth_failures_total counter",
                f"pcam_requests_auth_failures_total {self.auth_failures}",
                "# TYPE pcam_requests_rate_limited_total counter",
                f"pcam_requests_rate_limited_total {self.rate_limit_violations}",
                "# TYPE pcam_predictions_total counter",
                f"pcam_predictions_total {self.predictions}",
                "# TYPE pcam_request_latency_ms_avg gauge",
                f"pcam_request_latency_ms_avg {avg_latency:.4f}",
            ]
        )


metrics = MetricsStore()
rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SEC)
predictor: PCamPredictor | None = None


def _verify_api_key(credentials: Optional[HTTPAuthCredentials] = Depends(HTTPBearer(auto_error=False))) -> str:
    """
    Verify API key from Authorization header.
    
    SECURITY: Validates API key if enabled via environment variable.
    
    Args:
        credentials: HTTP Bearer token from Authorization header
    
    Returns:
        Client ID for rate limiting and logging
    
    Raises:
        HTTPException: If authentication fails
    """
    if not API_KEY_ENABLED:
        return "anonymous"
    
    if not credentials or not credentials.credentials:
        logger.warning("API request without credentials")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials != API_KEY:
        logger.warning(f"API request with invalid credentials from {credentials.credentials[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return "authenticated"


def _load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_predictor():
    deployment_config_path = os.getenv("DEPLOYMENT_CONFIG", "config/deployment_config.yaml")
    cfg = _load_yaml(deployment_config_path)

    # Log deployment configuration for reproducibility
    logger.info(f"Deployment configuration: {cfg}")

    threshold = float(cfg["api"].get("threshold", 0.5))
    use_tta = bool(cfg["api"].get("use_tta", False))

    if cfg.get("ensemble", {}).get("enabled"):
        return EnsemblePredictor(
            models=cfg["ensemble"]["members"],
            model_config_path=cfg["model"]["model_config"],
            data_config_path=cfg["model"]["data_config"],
            threshold=threshold,
            use_tta=use_tta,
        )

    return PCamPredictor(
        model_path=cfg["model"]["path"],
        model_config_path=cfg["model"]["model_config"],
        data_config_path=cfg["model"]["data_config"],
        threshold=threshold,
        use_tta=use_tta,
        pretrained_id=cfg["model"].get("pretrained_id"),
        registry_path=cfg["model"].get("pretrained_registry"),
        calibration_path=cfg["model"].get("calibration_path"),
    )


@app.on_event("startup")
def startup_event():
    global predictor
    predictor = _load_predictor()
    logger.info("Predictor loaded")


@app.get("/health")
def health():
    if predictor is None:
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return {"status": "ok", "model_loaded": True, "model_name": predictor.model_name}


@app.post("/predict")
async def predict(file: UploadFile = File(...), client_id: str = Depends(_verify_api_key)):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check rate limit
    if not rate_limiter.is_allowed(client_id):
        remaining = rate_limiter.get_remaining(client_id)
        metrics.record(0, 0, False, rate_limited=True)
        logger.warning(f"Rate limit exceeded for {client_id}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SEC}s",
        )

    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    error = False
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        result = predictor.predict(image)
        logger.info(f"Prediction successful: request_id={request_id}, client={client_id}")
        return JSONResponse(content={"request_id": request_id, **result})
    except Exception as exc:
        error = True
        logger.error(f"Prediction failed: {exc}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        latency_ms = (time.perf_counter() - start) * 1000.0
        metrics.record(latency_ms, predictions=1 if not error else 0, error=error)


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...), 
    client_id: str = Depends(_verify_api_key)
):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Check rate limit
    if not rate_limiter.is_allowed(client_id):
        metrics.record(0, 0, False, rate_limited=True)
        logger.warning(f"Rate limit exceeded for {client_id}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Max {RATE_LIMIT_REQUESTS} requests per {RATE_LIMIT_WINDOW_SEC}s",
        )

    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    error = False
    predictions = []

    try:
        for file in files:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            predictions.append(predictor.predict(image))
        logger.info(f"Batch prediction successful: request_id={request_id}, count={len(predictions)}")
        return JSONResponse(content={"request_id": request_id, "predictions": predictions})
    except Exception as exc:
        error = True
        logger.error(f"Batch prediction failed: {exc}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        latency_ms = (time.perf_counter() - start) * 1000.0
        metrics.record(latency_ms, predictions=len(predictions), error=error)


@app.get("/metrics")
def metrics_endpoint():
    return PlainTextResponse(metrics.render_prometheus())


# Seed random operations for reproducibility
import random
import numpy as np
from src.utils.reproducibility import set_seed

set_seed(42, deterministic=True, benchmark=False)

