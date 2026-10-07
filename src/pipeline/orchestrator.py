"""
Pipeline Orchestrator — Master pipeline that takes raw input and produces
a ThreatAssessment by coordinating all components of the DASI system.

Flow:
  raw_input → InputProcessor → Signal Analyzers (5) → TCAFE Fusion →
  CampaignManager → Alert Generation → ThreatAssessment

This is the single entry point for the entire system. External consumers
(API, CLI, dashboard) only need to call orchestrator.process(event).
"""

import os
import time
import json
import yaml
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np

from src.signals.base_signal import SignalOutput
from src.fusion.tcafe import TCAFE, ThreatAssessment
from src.campaign.campaign_manager import CampaignManager
from src.pipeline.input_processor import InputProcessor, IncidentInput

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result from one pipeline invocation."""
    
    incident_id: str
    session_id: str
    timestamp: float
    
    # Threat assessment from TCAFE
    threat_assessment: ThreatAssessment
    
    # Signal outputs (raw)
    signal_outputs: Dict[str, Dict]
    
    # Campaign info
    campaign_id: int
    campaign_alerts: List[Dict]
    
    # Processing metadata
    processing_time_ms: float
    signals_available: List[str]
    signals_missing: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        def _clean(val):
            if isinstance(val, dict):
                return {k: _clean(v) for k, v in val.items()}
            elif isinstance(val, (list, tuple)):
                return [_clean(v) for v in val]
            elif isinstance(val, (np.floating, float)):
                return float(val)
            elif isinstance(val, (np.integer, int)):
                return int(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, torch.Tensor):
                return val.detach().cpu().tolist()
            return val

        raw = {
            "incident_id": self.incident_id,
            "session_id": self.session_id,
            "timestamp": float(self.timestamp),
            "threat": self.threat_assessment.to_dict(),
            "signals": self.signal_outputs,
            "campaign_id": int(self.campaign_id),
            "alerts": self.campaign_alerts,
            "processing_time_ms": round(float(self.processing_time_ms), 2),
            "signals_available": self.signals_available,
            "signals_missing": self.signals_missing,
        }
        return _clean(raw)


class DASIOrchestrator:
    """Master pipeline orchestrator for the Digital Arrest Scam Intelligence System.
    
    Coordinates all components:
      1. Input validation and normalization
      2. Parallel signal analysis (5 analyzers)
      3. TCAFE temporal confidence-aware fusion
      4. Campaign lifecycle management
      5. Alert generation and routing
    
    Usage:
        orchestrator = DASIOrchestrator.from_config("config/model_config.yaml")
        result = orchestrator.process({"messages": [...], "caller_id": "..."})
        print(result.threat_assessment.threat_level)  # "HIGH"
    """
    
    SIGNAL_NAMES = [
        "conversation_semantics",
        "temporal_sequence",
        "voice_analysis",
        "network_graph",
        "geo_temporal",
    ]
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        system_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_config = model_config
        self.system_config = system_config or {}
        
        # Determine device
        device_str = self.system_config.get("system", {}).get("device", "auto")
        self.device = self._resolve_device(device_str)
        logger.info(f"DASI Orchestrator initializing on device: {self.device}")
        
        # Initialize components
        self.input_processor = InputProcessor(self.system_config.get("data", {}))
        
        # Initialize signal analyzers (lazy — loaded on first use)
        self._signal_analyzers = {}
        self._signals_initialized = False
        
        # Initialize TCAFE Fusion Engine
        tcafe_config = model_config.get("tcafe", {})
        self.tcafe = TCAFE(tcafe_config)
        self.tcafe.eval()
        logger.info("TCAFE Fusion Engine initialized.")
        
        # Initialize Campaign Manager
        campaign_config = model_config.get("campaign", {})
        self.campaign_manager = CampaignManager(campaign_config)
        logger.info("Campaign Manager initialized.")
        
        # Processing stats
        self._total_processed = 0
        self._total_scams_detected = 0
        self._processing_times: List[float] = []
    
    @classmethod
    def from_config(
        cls,
        model_config_path: str = "config/model_config.yaml",
        system_config_path: str = "config/system_config.yaml",
    ) -> "DASIOrchestrator":
        """Create orchestrator from config files.
        
        Args:
            model_config_path: Path to model configuration YAML.
            system_config_path: Path to system configuration YAML.
            
        Returns:
            Configured DASIOrchestrator instance.
        """
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)
        
        system_config = {}
        if os.path.exists(system_config_path):
            with open(system_config_path, "r") as f:
                system_config = yaml.safe_load(f)
        
        return cls(model_config, system_config)
    
    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)
    
    def _initialize_signals(self):
        """Lazily initialize signal analyzers.
        
        Imports are deferred to avoid loading heavy ML models
        until actually needed.
        """
        if self._signals_initialized:
            return
        
        try:
            from src.signals.conversation_semantics import ConversationSemanticsAnalyzer
            cs_config = self.model_config.get("conversation_semantics", {})
            self._signal_analyzers["conversation_semantics"] = (
                ConversationSemanticsAnalyzer(cs_config, device=str(self.device))
            )
            logger.info("  ✓ Conversation Semantics analyzer loaded.")
        except Exception as e:
            logger.warning(f"  ✗ Conversation Semantics failed to load: {e}")
        
        try:
            from src.signals.temporal_sequence import TemporalSequenceAnalyzer
            ts_config = self.model_config.get("temporal_sequence", {})
            self._signal_analyzers["temporal_sequence"] = (
                TemporalSequenceAnalyzer(ts_config, device=str(self.device))
            )
            logger.info("  ✓ Temporal Sequence analyzer loaded.")
        except Exception as e:
            logger.warning(f"  ✗ Temporal Sequence failed to load: {e}")
        
        try:
            from src.signals.voice_analysis import VoiceAnalysisAnalyzer
            va_config = self.model_config.get("voice_analysis", {})
            self._signal_analyzers["voice_analysis"] = (
                VoiceAnalysisAnalyzer(va_config, device=str(self.device))
            )
            logger.info("  ✓ Voice Analysis analyzer loaded.")
        except Exception as e:
            logger.warning(f"  ✗ Voice Analysis failed to load: {e}")
        
        try:
            from src.signals.network_graph import NetworkGraphAnalyzer
            ng_config = self.model_config.get("network_graph", {})
            self._signal_analyzers["network_graph"] = (
                NetworkGraphAnalyzer(ng_config, device=str(self.device))
            )
            logger.info("  ✓ Network Graph analyzer loaded.")
        except Exception as e:
            logger.warning(f"  ✗ Network Graph failed to load: {e}")
        
        try:
            from src.signals.geo_temporal import GeoTemporalAnalyzer
            gt_config = self.model_config.get("geo_temporal", {})
            self._signal_analyzers["geo_temporal"] = (
                GeoTemporalAnalyzer(gt_config, device=str(self.device))
            )
            logger.info("  ✓ Geo-Temporal analyzer loaded.")
        except Exception as e:
            logger.warning(f"  ✗ Geo-Temporal failed to load: {e}")
        
        self._signals_initialized = True
        logger.info(
            f"Signal analyzers ready: {len(self._signal_analyzers)}/{len(self.SIGNAL_NAMES)}"
        )
    
    def process(self, raw_data: Dict[str, Any]) -> PipelineResult:
        """Process a raw incident through the full DASI pipeline.
        
        This is the MAIN ENTRY POINT for the entire system.
        
        Args:
            raw_data: Raw incident data dictionary containing messages,
                      caller info, audio path, geo coordinates, etc.
                      
        Returns:
            PipelineResult with threat assessment, signal outputs,
            campaign assignment, and alerts.
        """
        start_time = time.time()
        
        # Step 0: Initialize signal analyzers (lazy)
        self._initialize_signals()
        
        # Step 1: Validate and normalize input
        incident = self.input_processor.process(raw_data)
        logger.info(f"Processing incident {incident.incident_id}")
        
        # Step 2: Run signal analyzers
        signal_outputs, available, missing = self._run_signal_analyzers(incident)
        
        # Step 3: TCAFE Fusion
        assessment = self.tcafe.fuse(
            signals=signal_outputs,
            incident_id=incident.incident_id,
            current_time=incident.timestamp,
            session_id=incident.session_id,
        )
        
        # Step 4: Campaign Management
        campaign_id, alerts = self.campaign_manager.process_incident(
            incident_id=incident.incident_id,
            threat_score=assessment.threat_score,
            fingerprint=assessment.fused_state,
            region=incident.region,
            escalation_speed=assessment.escalation_speed,
        )
        
        # Processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Update stats
        self._total_processed += 1
        if assessment.threat_score > 0.5:
            self._total_scams_detected += 1
        self._processing_times.append(processing_time)
        
        # Build signal output summaries
        signal_summaries = {}
        for s in signal_outputs:
            signal_summaries[s.signal_name] = s.to_dict()
        
        result = PipelineResult(
            incident_id=incident.incident_id,
            session_id=incident.session_id,
            timestamp=incident.timestamp,
            threat_assessment=assessment,
            signal_outputs=signal_summaries,
            campaign_id=campaign_id,
            campaign_alerts=[a.to_dict() for a in alerts],
            processing_time_ms=processing_time,
            signals_available=available,
            signals_missing=missing,
        )
        
        logger.info(
            f"Incident {incident.incident_id}: "
            f"threat={assessment.threat_score:.3f} ({assessment.threat_level}), "
            f"campaign=C-{campaign_id:03d}, "
            f"alerts={len(alerts)}, "
            f"time={processing_time:.1f}ms"
        )
        
        return result
    
    def _run_signal_analyzers(
        self, incident: IncidentInput
    ) -> Tuple[List[SignalOutput], List[str], List[str]]:
        """Run all available signal analyzers on the incident.
        
        Returns:
            Tuple of (signal_outputs, available_names, missing_names)
        """
        outputs = []
        available = []
        missing = []
        
        for name in self.SIGNAL_NAMES:
            if name not in self._signal_analyzers:
                # Generate a placeholder signal with low confidence
                outputs.append(self._generate_placeholder_signal(name, incident))
                missing.append(name)
                continue
            
            analyzer = self._signal_analyzers[name]
            try:
                data = self._prepare_signal_input(name, incident)
                output = analyzer.analyze(data)
                outputs.append(output)
                available.append(name)
            except Exception as e:
                logger.warning(f"Signal {name} analysis failed: {e}")
                outputs.append(self._generate_placeholder_signal(name, incident))
                missing.append(name)
        
        return outputs, available, missing
    
    def _prepare_signal_input(self, signal_name: str, incident: IncidentInput) -> Any:
        """Prepare signal-specific input from the normalized incident."""
        
        if signal_name == "conversation_semantics":
            # Concatenate messages for NLP analysis
            return {
                "messages": incident.messages,
                "language": incident.language,
            }
        
        elif signal_name == "temporal_sequence":
            # Extract temporal features
            return {
                "messages": incident.messages,
                "timestamps": [m.get("timestamp_offset", 0) for m in incident.messages],
            }
        
        elif signal_name == "voice_analysis":
            return {
                "audio_path": incident.audio_path,
                "audio_features": incident.audio_features,
            }
        
        elif signal_name == "network_graph":
            return {
                "caller_id": incident.caller_id,
                "callee_id": incident.callee_id,
                "metadata": incident.caller_metadata,
                "events": incident.network_events,
            }
        
        elif signal_name == "geo_temporal":
            return {
                "latitude": incident.latitude,
                "longitude": incident.longitude,
                "region": incident.region,
                "timestamp": incident.timestamp,
            }
        
        return {}
    
    def _generate_placeholder_signal(
        self, signal_name: str, incident: IncidentInput
    ) -> SignalOutput:
        """Generate a low-confidence placeholder when a signal analyzer is unavailable."""
        dim_map = {
            "conversation_semantics": 256,
            "temporal_sequence": 128,
            "voice_analysis": 128,
            "network_graph": 256,
            "geo_temporal": 64,
        }
        dim = dim_map.get(signal_name, 128)
        
        return SignalOutput(
            signal_name=signal_name,
            embedding=torch.randn(dim) * 0.01,  # Near-zero placeholder
            confidence=0.1,  # Very low confidence
            timestamp=incident.timestamp,
            metadata={"placeholder": True, "reason": "analyzer_unavailable"},
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get the overall system status."""
        avg_time = (
            np.mean(self._processing_times[-100:])
            if self._processing_times else 0.0
        )
        
        return {
            "status": "OPERATIONAL",
            "device": str(self.device),
            "signals_loaded": list(self._signal_analyzers.keys()),
            "signals_missing": [
                n for n in self.SIGNAL_NAMES if n not in self._signal_analyzers
            ],
            "total_processed": self._total_processed,
            "total_scams_detected": self._total_scams_detected,
            "detection_rate": (
                round(self._total_scams_detected / max(1, self._total_processed), 4)
            ),
            "avg_processing_time_ms": round(avg_time, 2),
            "active_campaigns": len(self.campaign_manager.get_active_campaigns()),
            "total_alerts": len(self.campaign_manager.alerts),
            "tcafe_status": self.tcafe.get_system_status(),
        }
    
    def get_campaign_dashboard(self) -> Dict[str, Any]:
        """Get campaign data for the dashboard."""
        return {
            "active_campaigns": self.campaign_manager.get_active_campaigns(),
            "recent_alerts": self.campaign_manager.get_recent_alerts(limit=20),
            "statistics": self.campaign_manager.get_statistics(),
        }
    
    def load_checkpoints(self, checkpoint_dir: str = "data/checkpoints"):
        """Load trained model checkpoints.
        
        Args:
            checkpoint_dir: Directory containing model checkpoints.
        """
        if not os.path.exists(checkpoint_dir):
            logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return
        
        # Load TCAFE weights
        tcafe_path = os.path.join(checkpoint_dir, "tcafe_best.pt")
        if os.path.exists(tcafe_path):
            state = torch.load(tcafe_path, map_location=self.device, weights_only=True)
            self.tcafe.load_state_dict(state["model_state_dict"])
            logger.info(f"TCAFE checkpoint loaded from {tcafe_path}")
        
        # Load individual signal checkpoints
        for name in self.SIGNAL_NAMES:
            ckpt_path = os.path.join(checkpoint_dir, f"{name}_best.pt")
            if os.path.exists(ckpt_path) and name in self._signal_analyzers:
                try:
                    state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                    analyzer = self._signal_analyzers[name]
                    if hasattr(analyzer, 'model'):
                        analyzer.model.load_state_dict(state["model_state_dict"])
                        logger.info(f"Checkpoint loaded for {name}")
                except Exception as e:
                    logger.warning(f"Failed to load checkpoint for {name}: {e}")
    
    def reset_session(self):
        """Reset session state while keeping models and campaigns."""
        self.tcafe.reset_session()
        self._processing_times.clear()
        logger.info("Session state reset.")
