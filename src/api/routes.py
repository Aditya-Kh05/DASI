"""
API Routes — FastAPI endpoint definitions for the DASI system.

Endpoints:
  POST   /analyze           — Analyze a single incident
  POST   /analyze/batch     — Analyze multiple incidents  
  GET    /campaigns         — List active campaigns
  GET    /campaigns/{id}    — Get specific campaign details
  GET    /alerts            — Get recent alerts
  GET    /status            — System status
  GET    /health            — Health check
  WS     /ws/live           — WebSocket for real-time streaming
"""

import time
import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query

from src.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    CampaignListResponse,
    CampaignResponse,
    AlertResponse,
    SystemStatusResponse,
    HealthResponse,
)

logger = logging.getLogger(__name__)

# Router — attached to the FastAPI app in app.py
router = APIRouter()

# Reference to orchestrator — set by app.py during lifespan
_orchestrator = None
_start_time = time.time()


def set_orchestrator(orchestrator):
    """Set the pipeline orchestrator for route handlers."""
    global _orchestrator
    _orchestrator = orchestrator


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        try:
            from src.pipeline.orchestrator import DASIOrchestrator
            _orchestrator = DASIOrchestrator.from_config()
            logger.info("Auto-initialized orchestrator after module reload.")
        except Exception as e:
            logger.error(f"Failed to auto-initialize orchestrator: {e}")
            raise HTTPException(
                status_code=503,
                detail="System not initialized. Please wait for startup to complete.",
            )
    return _orchestrator


# ════════════════════════════════════════════════════════
# Analysis Endpoints
# ════════════════════════════════════════════════════════

@router.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_incident(request: AnalyzeRequest):
    """Analyze a single incident for scam indicators.
    
    Runs the full DASI pipeline:
    1. Input validation and normalization
    2. 5 parallel signal analyzers
    3. TCAFE temporal confidence-aware fusion
    4. Campaign assignment and alert generation
    
    Returns a complete threat assessment with signal contributions,
    escalation velocity, and campaign context.
    """
    orchestrator = _get_orchestrator()
    
    # Convert Pydantic model to raw dict for the orchestrator
    raw_data = {
        "messages": [m.model_dump() for m in request.messages],
        "incident_id": request.incident_id,
        "session_id": request.session_id,
        "language": request.language,
        "audio_path": request.audio_path,
        "caller_id": request.caller_id,
        "callee_id": request.callee_id,
        "latitude": request.latitude,
        "longitude": request.longitude,
        "region": request.region,
        "network_events": request.network_events,
        "caller_metadata": request.caller_metadata,
    }
    
    try:
        # Run pipeline (CPU-bound — run in thread pool)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, orchestrator.process, raw_data)
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/analyze/batch", tags=["Analysis"])
async def analyze_batch(request: BatchAnalyzeRequest):
    """Analyze multiple incidents in batch.
    
    Processes up to 100 incidents sequentially and returns
    all threat assessments.
    """
    orchestrator = _get_orchestrator()
    results = []
    errors = []
    
    for i, incident in enumerate(request.incidents):
        try:
            raw_data = {
                "messages": [m.model_dump() for m in incident.messages],
                "incident_id": incident.incident_id,
                "session_id": incident.session_id,
                "language": incident.language,
                "caller_id": incident.caller_id,
                "region": incident.region,
            }
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, orchestrator.process, raw_data)
            results.append(result.to_dict())
        except Exception as e:
            errors.append({"index": i, "error": str(e)})
    
    return {
        "results": results,
        "errors": errors,
        "total": len(request.incidents),
        "successful": len(results),
        "failed": len(errors),
    }


# ════════════════════════════════════════════════════════
# Campaign Endpoints
# ════════════════════════════════════════════════════════

@router.get("/campaigns", response_model=CampaignListResponse, tags=["Campaigns"])
async def list_campaigns():
    """Get all active scam campaigns.
    
    Returns campaigns with their incident counts, severity,
    geographic spread, and alert history.
    """
    orchestrator = _get_orchestrator()
    stats = orchestrator.campaign_manager.get_statistics()
    active = orchestrator.campaign_manager.get_active_campaigns()
    
    return {
        "active_campaigns": active,
        "total_campaigns": stats["total_campaigns"],
        "total_incidents": stats["total_incidents"],
    }


@router.get("/campaigns/{campaign_id}", tags=["Campaigns"])
async def get_campaign(campaign_id: int):
    """Get details for a specific campaign."""
    orchestrator = _get_orchestrator()
    campaign = orchestrator.campaign_manager.get_campaign(campaign_id)
    
    if campaign is None:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found.")
    
    return campaign


# ════════════════════════════════════════════════════════
# Alert Endpoints
# ════════════════════════════════════════════════════════

@router.get("/alerts", tags=["Alerts"])
async def list_alerts(
    limit: int = Query(50, ge=1, le=200, description="Max alerts to return"),
):
    """Get recent alerts sorted by timestamp (newest first)."""
    orchestrator = _get_orchestrator()
    alerts = orchestrator.campaign_manager.get_recent_alerts(limit=limit)
    return {"alerts": alerts, "total": len(alerts)}


# ════════════════════════════════════════════════════════
# System Endpoints
# ════════════════════════════════════════════════════════

@router.get("/status", response_model=SystemStatusResponse, tags=["System"])
async def system_status():
    """Get the current system status.
    
    Includes signal analyzer health, processing stats,
    and campaign intelligence metrics.
    """
    orchestrator = _get_orchestrator()
    status = orchestrator.get_system_status()
    return status


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "uptime_seconds": round(time.time() - _start_time, 2),
    }


# ════════════════════════════════════════════════════════
# WebSocket — Real-Time Streaming
# ════════════════════════════════════════════════════════

class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Active: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Active: {len(self.active_connections)}")
    
    async def broadcast(self, data: dict):
        for conn in self.active_connections:
            try:
                await conn.send_json(data)
            except Exception:
                pass


ws_manager = ConnectionManager()


@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """WebSocket endpoint for real-time threat monitoring.
    
    Clients receive:
      - New threat assessments as they are computed
      - Campaign updates and alerts
      - System health heartbeats every 30s
    """
    await ws_manager.connect(websocket)
    
    try:
        while True:
            # Receive analysis requests via WebSocket
            data = await websocket.receive_json()
            
            if data.get("type") == "analyze":
                orchestrator = _get_orchestrator()
                try:
                    result = orchestrator.process(data.get("payload", {}))
                    await websocket.send_json({
                        "type": "threat_assessment",
                        "data": result.to_dict(),
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e),
                    })
            
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
