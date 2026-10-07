"""
Pydantic Schemas — Request/Response models for the DASI REST API.

All API endpoints use these schemas for validation and serialization.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ════════════════════════════════════════════════════════
# Request Schemas
# ════════════════════════════════════════════════════════

class MessageSchema(BaseModel):
    """A single message in a conversation."""
    role: str = Field(..., description="Speaker role: 'caller', 'scammer', 'victim', 'unknown'")
    text: str = Field(..., description="Message content text")
    timestamp_offset: float = Field(0.0, description="Seconds offset from call start")


class AnalyzeRequest(BaseModel):
    """Request body for POST /analyze endpoint."""
    
    # Required
    messages: List[MessageSchema] = Field(
        ..., description="Conversation messages to analyze", min_length=1
    )
    
    # Optional identifiers
    incident_id: Optional[str] = Field(None, description="Custom incident ID")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn tracking")
    
    # Optional signals
    language: Optional[str] = Field(None, description="Language code: 'hi', 'en', 'hi-en'")
    audio_path: Optional[str] = Field(None, description="Path to audio file for voice analysis")
    caller_id: Optional[str] = Field(None, description="Caller phone/device identifier")
    callee_id: Optional[str] = Field(None, description="Callee phone/device identifier")
    
    # Optional geo data
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    region: Optional[str] = Field(None, description="Geographic region name")
    
    # Network events
    network_events: List[Dict[str, Any]] = Field(
        default_factory=list, description="Communication network events"
    )
    caller_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional caller metadata"
    )

    model_config = {"json_schema_extra": {
        "examples": [{
            "messages": [
                {"role": "caller", "text": "This is CBI officer Sharma. You are under investigation for money laundering.", "timestamp_offset": 0},
                {"role": "victim", "text": "What? I haven't done anything wrong!", "timestamp_offset": 5},
                {"role": "caller", "text": "Your Aadhaar has been linked to suspicious transactions. You must transfer ₹5,00,000 immediately or face arrest.", "timestamp_offset": 12},
            ],
            "language": "en",
            "region": "Delhi NCR",
        }]
    }}


class BatchAnalyzeRequest(BaseModel):
    """Request body for POST /analyze/batch endpoint."""
    incidents: List[AnalyzeRequest] = Field(
        ..., description="List of incidents to analyze", min_length=1, max_length=100
    )


# ════════════════════════════════════════════════════════
# Response Schemas
# ════════════════════════════════════════════════════════

class ThreatResponse(BaseModel):
    """Threat assessment from TCAFE fusion engine."""
    threat_score: float = Field(..., description="Threat level [0, 1]")
    threat_level: str = Field(..., description="MINIMAL, LOW, MEDIUM, HIGH, CRITICAL")
    escalation_speed: float = Field(..., description="Rate of threat change")
    escalation_trend: str = Field(..., description="ESCALATING, STABLE, DE_ESCALATING")
    escalation_alert: str = Field(..., description="Alert level for escalation")
    campaign_id: int = Field(..., description="Assigned campaign ID (-1 if none)")
    is_new_campaign: bool = Field(..., description="Whether a new campaign was created")
    campaign_risk: float = Field(..., description="Campaign-level risk score")
    signal_contributions: Dict[str, float] = Field(..., description="Per-signal contribution weights")
    confidence_decay: Dict[str, Dict] = Field(..., description="Confidence decay per signal")
    cross_correlations: Dict[str, Dict] = Field(..., description="Cross-signal correlations")
    timestamp: float
    processing_time_ms: float


class AlertResponse(BaseModel):
    """Campaign alert."""
    alert_id: str
    timestamp: float
    alert_type: str  # EARLY_WARNING, ESCALATION, NEW_CAMPAIGN, CAMPAIGN_SPREAD
    severity: str    # LOW, MEDIUM, HIGH, CRITICAL
    campaign_id: Optional[int]
    incident_id: Optional[str]
    message: str


class AnalyzeResponse(BaseModel):
    """Response from POST /analyze endpoint."""
    incident_id: str
    session_id: str
    timestamp: float
    threat: ThreatResponse
    signals: Dict[str, Dict[str, Any]]
    campaign_id: int
    alerts: List[AlertResponse]
    processing_time_ms: float
    signals_available: List[str]
    signals_missing: List[str]


class CampaignResponse(BaseModel):
    """Campaign details."""
    campaign_id: int
    status: str
    incident_count: int
    current_severity: float
    peak_severity: float
    avg_severity: float
    duration_hours: float
    regions: List[str]
    num_alerts: int
    created_at: float
    updated_at: float


class CampaignListResponse(BaseModel):
    """Response from GET /campaigns endpoint."""
    active_campaigns: List[CampaignResponse]
    total_campaigns: int
    total_incidents: int


class SystemStatusResponse(BaseModel):
    """Response from GET /status endpoint."""
    status: str
    device: str
    signals_loaded: List[str]
    signals_missing: List[str]
    total_processed: int
    total_scams_detected: int
    detection_rate: float
    avg_processing_time_ms: float
    active_campaigns: int
    total_alerts: int


class HealthResponse(BaseModel):
    """Response from GET /health endpoint."""
    status: str = "healthy"
    version: str
    uptime_seconds: float
