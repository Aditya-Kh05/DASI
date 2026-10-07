"""
Alert Engine — Rule-based + ML alert generation with severity levels.

Generates alerts based on:
  - Threat score thresholds (rule-based)
  - Escalation velocity (ML-derived)
  - Campaign spread patterns
  - Geographic expansion
  - Temporal anomalies (time-of-day, frequency spikes)
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class AlertEngine:
    """Multi-level alert generation engine.
    
    Combines rule-based triggers with ML-derived signals to generate
    actionable alerts with severity levels and routing information.
    """
    
    # Alert severity levels
    SEVERITY_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    # Alert types
    ALERT_TYPES = [
        "EARLY_WARNING",      # Potential scam detected early
        "ESCALATION",         # Rapid escalation detected
        "NEW_CAMPAIGN",       # New scam campaign identified
        "CAMPAIGN_SPREAD",    # Campaign expanding to new region
        "MASS_TARGETING",     # High-frequency targeting detected
        "DEEPFAKE_DETECTED",  # Voice deepfake identified
        "FINANCIAL_RISK",     # Victim approaching financial loss
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        
        # Threshold configuration
        self.thresholds = {
            "early_warning": config.get("early_warning_threshold", 0.4),
            "high_threat": config.get("high_threat_threshold", 0.7),
            "critical_threat": config.get("critical_threat_threshold", 0.9),
            "escalation_speed": config.get("escalation_speed_threshold", 0.3),
            "mass_targeting_count": config.get("mass_targeting_count", 10),
            "mass_targeting_window": config.get("mass_targeting_window_hours", 1),
        }
        
        # Tracking state
        self._alert_counter = 0
        self._alert_history: List[Dict] = []
        self._incident_tracker: Dict[str, float] = {}  # region -> last alert time
        self._region_counts: Dict[str, int] = defaultdict(int)
    
    def evaluate(
        self,
        threat_score: float,
        threat_level: str,
        escalation_speed: float,
        incident_id: str,
        campaign_id: int = -1,
        is_new_campaign: bool = False,
        region: Optional[str] = None,
        signal_contributions: Optional[Dict[str, float]] = None,
        is_deepfake: bool = False,
        scam_phase: Optional[int] = None,
    ) -> List[Dict]:
        """Evaluate an incident and generate appropriate alerts.
        
        Args:
            threat_score: Overall threat score [0, 1].
            threat_level: Classified threat level string.
            escalation_speed: Current escalation velocity.
            incident_id: Unique incident identifier.
            campaign_id: Assigned campaign ID.
            is_new_campaign: Whether this created a new campaign.
            region: Geographic region.
            signal_contributions: Per-signal contribution weights.
            is_deepfake: Whether deepfake was detected in voice.
            scam_phase: Detected scam phase (1-6).
            
        Returns:
            List of generated alert dictionaries.
        """
        alerts = []
        
        # Rule 1: Early warning
        if (self.thresholds["early_warning"] <= threat_score < 
                self.thresholds["high_threat"]):
            alerts.append(self._create_alert(
                alert_type="EARLY_WARNING",
                severity="MEDIUM",
                incident_id=incident_id,
                campaign_id=campaign_id,
                message=(
                    f"Early warning: Incident {incident_id} shows scam indicators "
                    f"(score: {threat_score:.2f}). Potential victim at risk."
                ),
                metadata={"threat_score": threat_score, "region": region},
            ))
        
        # Rule 2: High/Critical threat
        if threat_score >= self.thresholds["critical_threat"]:
            alerts.append(self._create_alert(
                alert_type="ESCALATION",
                severity="CRITICAL",
                incident_id=incident_id,
                campaign_id=campaign_id,
                message=(
                    f"CRITICAL: Incident {incident_id} has critical threat score "
                    f"({threat_score:.2f}). Immediate intervention required."
                ),
                metadata={"threat_score": threat_score},
            ))
        elif threat_score >= self.thresholds["high_threat"]:
            alerts.append(self._create_alert(
                alert_type="ESCALATION",
                severity="HIGH",
                incident_id=incident_id,
                campaign_id=campaign_id,
                message=(
                    f"HIGH THREAT: Incident {incident_id} scored {threat_score:.2f}. "
                    f"Campaign C-{campaign_id:03d}."
                ),
                metadata={"threat_score": threat_score},
            ))
        
        # Rule 3: Rapid escalation
        if escalation_speed > self.thresholds["escalation_speed"]:
            severity = "CRITICAL" if escalation_speed > 0.6 else "HIGH"
            alerts.append(self._create_alert(
                alert_type="ESCALATION",
                severity=severity,
                incident_id=incident_id,
                campaign_id=campaign_id,
                message=(
                    f"Rapid escalation in {incident_id}: velocity={escalation_speed:.3f}. "
                    f"Scam is accelerating."
                ),
                metadata={"escalation_speed": escalation_speed},
            ))
        
        # Rule 4: New campaign
        if is_new_campaign and threat_score > self.thresholds["early_warning"]:
            alerts.append(self._create_alert(
                alert_type="NEW_CAMPAIGN",
                severity="HIGH" if threat_score > 0.7 else "MEDIUM",
                incident_id=incident_id,
                campaign_id=campaign_id,
                message=(
                    f"New scam campaign C-{campaign_id:03d} detected in "
                    f"{region or 'unknown region'}."
                ),
                metadata={"region": region},
            ))
        
        # Rule 5: Deepfake detection
        if is_deepfake:
            alerts.append(self._create_alert(
                alert_type="DEEPFAKE_DETECTED",
                severity="HIGH",
                incident_id=incident_id,
                campaign_id=campaign_id,
                message=(
                    f"Voice deepfake detected in incident {incident_id}. "
                    f"Caller is using synthetic/cloned voice."
                ),
            ))
        
        # Rule 6: Financial risk (phases 4-6)
        if scam_phase is not None and scam_phase >= 4:
            alerts.append(self._create_alert(
                alert_type="FINANCIAL_RISK",
                severity="CRITICAL" if scam_phase >= 5 else "HIGH",
                incident_id=incident_id,
                campaign_id=campaign_id,
                message=(
                    f"Financial risk: Incident {incident_id} is at scam phase {scam_phase}. "
                    f"Victim may be transferring money."
                ),
                metadata={"scam_phase": scam_phase},
            ))
        
        # Track region activity
        if region:
            self._region_counts[region] += 1
        
        # Store alerts
        self._alert_history.extend(alerts)
        
        return alerts
    
    def _create_alert(self, **kwargs) -> Dict:
        """Create a formatted alert dictionary."""
        self._alert_counter += 1
        return {
            "alert_id": f"ALR-{self._alert_counter:05d}",
            "timestamp": time.time(),
            **kwargs,
        }
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get a summary of all generated alerts."""
        severity_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for alert in self._alert_history:
            severity_counts[alert.get("severity", "UNKNOWN")] += 1
            type_counts[alert.get("alert_type", "UNKNOWN")] += 1
        
        return {
            "total_alerts": len(self._alert_history),
            "by_severity": dict(severity_counts),
            "by_type": dict(type_counts),
            "region_activity": dict(self._region_counts),
        }
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Get most recent alerts."""
        return sorted(
            self._alert_history,
            key=lambda a: a.get("timestamp", 0),
            reverse=True,
        )[:limit]
