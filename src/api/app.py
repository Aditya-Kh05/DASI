"""
FastAPI Application — DASI REST API Server.

Provides a REST interface to the Digital Arrest Scam Intelligence System.
Run with: uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

Features:
  - CORS support for dashboard connections
  - Lifespan events for startup/shutdown
  - Automatic Swagger/OpenAPI documentation at /docs
  - Request rate limiting (production)
"""

import os
import time
import logging
import warnings
from contextlib import asynccontextmanager

# Suppress PyTorch internal prototype notices from spamming console/terminal
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", message=".*nested tensors is in prototype stage.*")

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, set_orchestrator
from src.pipeline.orchestrator import DASIOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — initialize and tear down the pipeline."""
    logger.info("=" * 60)
    logger.info("  DASI — Digital Arrest Scam Intelligence System")
    logger.info("  Starting API server...")
    logger.info("=" * 60)
    
    # Load configuration
    model_config_path = os.environ.get("DASI_MODEL_CONFIG", "config/model_config.yaml")
    system_config_path = os.environ.get("DASI_SYSTEM_CONFIG", "config/system_config.yaml")
    
    try:
        orchestrator = DASIOrchestrator.from_config(model_config_path, system_config_path)
        
        # Load checkpoints if available
        checkpoint_dir = os.environ.get("DASI_CHECKPOINTS", "data/checkpoints")
        orchestrator.load_checkpoints(checkpoint_dir)
        
        # Inject orchestrator into routes
        set_orchestrator(orchestrator)
        app.state.orchestrator = orchestrator
        
        logger.info("DASI system fully initialized and ready.")
        logger.info(f"API docs available at: http://localhost:8000/docs")
    except Exception as e:
        logger.error(f"Failed to initialize DASI: {e}", exc_info=True)
        raise
    
    yield  # Application running
    
    # Shutdown
    logger.info("DASI system shutting down...")


# Create FastAPI app
app = FastAPI(
    title="DASI — Digital Arrest Scam Intelligence System",
    description=(
        "AI-powered multi-signal fusion system for detecting and tracking "
        "'digital arrest' scam campaigns in India using the novel TCAFE "
        "(Temporal Confidence-Aware Fusion Engine) architecture.\n\n"
        "## Signals Analyzed\n"
        "1. **Conversation Semantics** — NLP analysis of scam language patterns\n"
        "2. **Temporal Sequence** — Phase progression tracking\n"
        "3. **Voice Analysis** — Deepfake detection + speaker profiling\n"
        "4. **Network Graph** — Communication graph topology analysis\n"
        "5. **Geo-Temporal** — Geographic spread pattern detection\n\n"
        "## Core Innovation\n"
        "TCAFE fuses all 5 signals with temporal confidence decay, "
        "cross-signal attention, escalation velocity tracking, and "
        "campaign fingerprinting — going far beyond simple weighted fusion."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow dashboard and development origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:3000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:3000",
        "*",  # Development only — restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes
app.include_router(router, prefix="/api/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    tb = traceback.format_exc()
    logger.error(f"Global server error on {request.url.path}: {exc}\n{tb}")
    os.makedirs("data", exist_ok=True)
    with open("data/last_error.log", "w", encoding="utf-8") as f:
        f.write(f"Exception: {exc}\n\nTraceback:\n{tb}")
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={"error": str(exc), "traceback": tb})


# Root redirect
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — API information."""
    return {
        "name": "DASI — Digital Arrest Scam Intelligence System",
        "version": "0.1.0",
        "docs": "/docs",
        "api_prefix": "/api/v1",
        "endpoints": {
            "analyze": "POST /api/v1/analyze",
            "campaigns": "GET /api/v1/campaigns",
            "alerts": "GET /api/v1/alerts",
            "status": "GET /api/v1/status",
            "health": "GET /api/v1/health",
            "websocket": "WS /api/v1/ws/live",
        },
    }
