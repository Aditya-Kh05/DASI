"""
Input Processor — Validates and normalizes incoming data for the DASI pipeline.

Accepts raw input from API calls or CLI, validates required fields,
normalizes text encoding, resolves file paths, and produces a clean
IncidentInput that the Orchestrator can process.
"""

import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class IncidentInput:
    """Validated, normalized input for the DASI pipeline.
    
    This is the single interface between external data sources
    and the internal analysis pipeline.
    """
    
    # Unique identifiers
    incident_id: str
    session_id: str
    timestamp: float
    
    # Text signals (for Conversation Semantics + Temporal Sequence)
    messages: List[Dict[str, str]]  # [{role, text, timestamp_offset}]
    language: str                    # "hi", "en", "hi-en" (code-mixed)
    
    # Audio signals (for Voice Analysis)
    audio_path: Optional[str] = None
    audio_features: Optional[Dict] = None  # Pre-extracted features
    
    # Caller metadata (for Network Graph)
    caller_id: Optional[str] = None
    callee_id: Optional[str] = None
    caller_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Network events
    network_events: List[Dict] = field(default_factory=list)
    
    # Geographic data (for Geo-Temporal)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    region: Optional[str] = None
    
    # Labels (for training/evaluation)
    is_scam: Optional[bool] = None
    scam_phase: Optional[int] = None
    campaign_label: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "num_messages": len(self.messages),
            "language": self.language,
            "has_audio": self.audio_path is not None,
            "has_network": len(self.network_events) > 0,
            "has_geo": self.latitude is not None,
            "region": self.region,
        }


class InputProcessor:
    """Validates and normalizes raw input data.
    
    Responsibilities:
      - Validate required fields
      - Normalize text encoding (Hindi/English/code-mixed)
      - Resolve and verify audio file paths
      - Generate incident IDs if not provided
      - Sanitize potentially malicious input
    """
    
    SUPPORTED_LANGUAGES = {"hi", "en", "hi-en", "mr", "ta", "te", "bn"}
    MAX_MESSAGE_LENGTH = 5000
    MAX_MESSAGES = 500
    
    INDIAN_REGIONS = [
        "Delhi NCR", "Mumbai Metropolitan", "Bangalore Urban",
        "Chennai Metropolitan", "Kolkata Metropolitan", "Hyderabad",
        "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Chandigarh",
        "Bhopal", "Patna", "Thiruvananthapuram", "Guwahati",
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._active_sessions: Dict[str, float] = {}
    
    def process(self, raw_data: Dict[str, Any]) -> IncidentInput:
        """Process and validate raw input into an IncidentInput.
        
        Args:
            raw_data: Dictionary with raw input fields.
            
        Returns:
            Validated IncidentInput.
            
        Raises:
            ValueError: If required fields are missing or invalid.
        """
        # Generate IDs
        incident_id = raw_data.get("incident_id") or f"INC-{uuid.uuid4().hex[:8].upper()}"
        session_id = raw_data.get("session_id") or f"SES-{uuid.uuid4().hex[:8].upper()}"
        timestamp = raw_data.get("timestamp") or time.time()
        
        # Validate and normalize messages
        messages = self._normalize_messages(raw_data.get("messages", []))
        if not messages:
            raise ValueError("At least one message is required for analysis.")
        
        # Detect language
        language = raw_data.get("language") or self._detect_language(messages)
        
        # Validate audio
        audio_path = self._validate_audio(raw_data.get("audio_path"))
        audio_features = raw_data.get("audio_features")
        
        # Validate geographic data
        latitude, longitude, region = self._validate_geo(
            raw_data.get("latitude"),
            raw_data.get("longitude"),
            raw_data.get("region"),
        )
        
        # Caller metadata
        caller_id = self._sanitize_id(raw_data.get("caller_id"))
        callee_id = self._sanitize_id(raw_data.get("callee_id"))
        caller_metadata = raw_data.get("caller_metadata", {})
        
        # Network events
        network_events = raw_data.get("network_events", [])
        
        # Labels (optional, for training)
        is_scam = raw_data.get("is_scam")
        scam_phase = raw_data.get("scam_phase")
        campaign_label = raw_data.get("campaign_label")
        
        # Track session
        self._active_sessions[session_id] = timestamp
        
        return IncidentInput(
            incident_id=incident_id,
            session_id=session_id,
            timestamp=timestamp,
            messages=messages,
            language=language,
            audio_path=audio_path,
            audio_features=audio_features,
            caller_id=caller_id,
            callee_id=callee_id,
            caller_metadata=caller_metadata,
            network_events=network_events,
            latitude=latitude,
            longitude=longitude,
            region=region,
            is_scam=is_scam,
            scam_phase=scam_phase,
            campaign_label=campaign_label,
        )
    
    def _normalize_messages(self, messages: List[Any]) -> List[Dict[str, str]]:
        """Normalize and validate message list."""
        if not messages:
            return []
        
        normalized = []
        for i, msg in enumerate(messages[:self.MAX_MESSAGES]):
            if isinstance(msg, str):
                msg = {"role": "unknown", "text": msg}
            elif not isinstance(msg, dict):
                continue
            
            text = str(msg.get("text", "")).strip()
            if not text:
                continue
            
            # Truncate overly long messages
            if len(text) > self.MAX_MESSAGE_LENGTH:
                text = text[:self.MAX_MESSAGE_LENGTH] + "..."
            
            # Sanitize — remove potential injection/control chars
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
            
            normalized.append({
                "role": str(msg.get("role", "unknown")).lower(),
                "text": text,
                "timestamp_offset": float(msg.get("timestamp_offset", i * 10)),
            })
        
        return normalized
    
    def _detect_language(self, messages: List[Dict]) -> str:
        """Basic language detection based on character scripts."""
        all_text = " ".join(m["text"] for m in messages)
        
        # Count Devanagari vs Latin characters
        devanagari = len(re.findall(r'[\u0900-\u097F]', all_text))
        latin = len(re.findall(r'[a-zA-Z]', all_text))
        total = devanagari + latin or 1
        
        hindi_ratio = devanagari / total
        if hindi_ratio > 0.6:
            return "hi"
        elif hindi_ratio > 0.2:
            return "hi-en"
        return "en"
    
    def _validate_audio(self, audio_path: Optional[str]) -> Optional[str]:
        """Validate and resolve audio file path."""
        if audio_path is None:
            return None
        
        audio_path = str(audio_path).strip()
        if not audio_path:
            return None
        
        # Resolve relative paths
        if not os.path.isabs(audio_path):
            base = self.config.get("audio_base_dir", "data/audio")
            audio_path = os.path.join(base, audio_path)
        
        # Check file exists (non-blocking — return path even if missing)
        if not os.path.exists(audio_path):
            # File might be on a remote path or will be provided later
            pass
        
        return audio_path
    
    def _validate_geo(
        self,
        lat: Optional[float],
        lon: Optional[float],
        region: Optional[str],
    ) -> tuple:
        """Validate geographic coordinates."""
        if lat is not None and lon is not None:
            try:
                lat = float(lat)
                lon = float(lon)
                # Validate India bounds (rough)
                if not (6.0 <= lat <= 37.0 and 68.0 <= lon <= 98.0):
                    # Outside India — still accept but flag
                    pass
            except (ValueError, TypeError):
                lat, lon = None, None
        
        if region and region not in self.INDIAN_REGIONS:
            # Accept custom regions
            pass
        
        return lat, lon, region
    
    def _sanitize_id(self, id_value: Optional[str]) -> Optional[str]:
        """Sanitize identifier strings."""
        if id_value is None:
            return None
        return re.sub(r'[^a-zA-Z0-9_\-+.]', '', str(id_value))[:64]
    
    def get_active_sessions(self) -> Dict[str, float]:
        """Return currently tracked sessions."""
        return dict(self._active_sessions)
