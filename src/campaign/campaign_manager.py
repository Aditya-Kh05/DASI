"""
Campaign Manager — Lifecycle management for detected scam campaigns.

Handles campaign creation, updating, merging, expiration, and alerting.
Each campaign tracks a group of related scam incidents that share
similar temporal trajectories, network topologies, or geographic patterns.
"""

import time
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import numpy as np


@dataclass
class Campaign:
    """Represents a tracked scam campaign."""
    campaign_id: int
    created_at: float
    updated_at: float
    
    # Incidents
    incident_ids: List[str] = field(default_factory=list)
    incident_count: int = 0
    
    # Severity tracking
    severity_scores: List[float] = field(default_factory=list)
    current_severity: float = 0.0
    peak_severity: float = 0.0
    
    # Fingerprint
    fingerprint: Optional[torch.Tensor] = None
    
    # Geographic spread
    regions: List[str] = field(default_factory=list)
    
    # Status
    status: str = "ACTIVE"  # ACTIVE, DORMANT, EXPIRED, NEUTRALIZED
    
    # Alert history
    alerts_sent: List[Dict] = field(default_factory=list)
    
    @property
    def duration_hours(self) -> float:
        return (self.updated_at - self.created_at) / 3600
    
    @property
    def avg_severity(self) -> float:
        return np.mean(self.severity_scores) if self.severity_scores else 0.0
    
    def to_dict(self) -> Dict:
        return {
            "campaign_id": self.campaign_id,
            "status": self.status,
            "incident_count": self.incident_count,
            "current_severity": round(self.current_severity, 4),
            "peak_severity": round(self.peak_severity, 4),
            "avg_severity": round(self.avg_severity, 4),
            "duration_hours": round(self.duration_hours, 2),
            "regions": list(set(self.regions)),
            "num_alerts": len(self.alerts_sent),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class Alert:
    """A generated alert for a campaign or incident."""
    alert_id: str
    timestamp: float
    alert_type: str       # "EARLY_WARNING", "ESCALATION", "NEW_CAMPAIGN", "CAMPAIGN_SPREAD"
    severity: str         # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    campaign_id: Optional[int]
    incident_id: Optional[str]
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "campaign_id": self.campaign_id,
            "incident_id": self.incident_id,
            "message": self.message,
        }


class CampaignManager:
    """Manages the lifecycle of detected scam campaigns.
    
    Responsibilities:
      - Create new campaigns from unmatched fingerprints
      - Update existing campaigns with new incidents
      - Merge campaigns when fingerprints converge
      - Expire dormant campaigns
      - Generate alerts based on campaign state changes
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.campaigns: Dict[int, Campaign] = {}
        self.alerts: List[Alert] = []
        self._next_campaign_id = 0
        self._next_alert_id = 0
        
        # Config parameters
        self.max_campaigns = config.get("max_active_campaigns", 1000)
        self.expiry_hours = config.get("campaign_expiry_hours", 168)
        self.similarity_threshold = config.get("fingerprint_similarity_threshold", 0.75)
        
        self.alert_levels = config.get("alert_levels", {
            "low": 0.3, "medium": 0.5, "high": 0.7, "critical": 0.9,
        })
    
    def process_incident(
        self,
        incident_id: str,
        threat_score: float,
        fingerprint: Optional[torch.Tensor] = None,
        region: Optional[str] = None,
        escalation_speed: float = 0.0,
    ) -> Tuple[int, List[Alert]]:
        """Process a new incident and assign to a campaign.
        
        Args:
            incident_id: Unique identifier for this incident.
            threat_score: TCAFE threat score [0, 1].
            fingerprint: Campaign fingerprint embedding.
            region: Geographic region of the incident.
            escalation_speed: Current escalation velocity.
        
        Returns:
            Tuple of (campaign_id, list of new alerts).
        """
        now = time.time()
        new_alerts = []
        
        # Expire old campaigns
        self._expire_campaigns(now)
        
        # Try to match to existing campaign
        matched_id = self._match_fingerprint(fingerprint) if fingerprint is not None else None
        
        if matched_id is not None:
            # Update existing campaign
            campaign = self.campaigns[matched_id]
            campaign.incident_ids.append(incident_id)
            campaign.incident_count += 1
            campaign.severity_scores.append(threat_score)
            campaign.current_severity = threat_score
            campaign.peak_severity = max(campaign.peak_severity, threat_score)
            campaign.updated_at = now
            
            if region and region not in campaign.regions:
                campaign.regions.append(region)
                # Alert: campaign spreading to new region
                alert = self._create_alert(
                    alert_type="CAMPAIGN_SPREAD",
                    severity=self._get_alert_severity(threat_score),
                    campaign_id=matched_id,
                    incident_id=incident_id,
                    message=f"Campaign C-{matched_id:03d} has spread to {region}. "
                            f"Now active in {len(campaign.regions)} regions.",
                )
                new_alerts.append(alert)
            
            # Update fingerprint (exponential moving average)
            if fingerprint is not None and campaign.fingerprint is not None:
                alpha = 0.3
                campaign.fingerprint = alpha * fingerprint + (1 - alpha) * campaign.fingerprint
            
        else:
            # Create new campaign
            matched_id = self._create_campaign(incident_id, threat_score, fingerprint, region, now)
            
            if threat_score > self.alert_levels.get("low", 0.3):
                alert = self._create_alert(
                    alert_type="NEW_CAMPAIGN",
                    severity=self._get_alert_severity(threat_score),
                    campaign_id=matched_id,
                    incident_id=incident_id,
                    message=f"New scam campaign C-{matched_id:03d} detected "
                            f"in {region or 'unknown region'}. Initial severity: {threat_score:.2f}.",
                )
                new_alerts.append(alert)
        
        # Check for escalation alerts
        if escalation_speed > 0.3:
            alert = self._create_alert(
                alert_type="ESCALATION",
                severity="HIGH" if escalation_speed > 0.5 else "MEDIUM",
                campaign_id=matched_id,
                incident_id=incident_id,
                message=f"Rapid escalation detected in incident {incident_id}. "
                        f"Velocity: {escalation_speed:.3f}. Immediate attention required.",
            )
            new_alerts.append(alert)
        
        # Check for early warning
        if threat_score > 0.4 and threat_score < 0.7:
            alert = self._create_alert(
                alert_type="EARLY_WARNING",
                severity="MEDIUM",
                campaign_id=matched_id,
                incident_id=incident_id,
                message=f"Early warning: Incident {incident_id} shows scam indicators "
                        f"(score: {threat_score:.2f}). Potential victim at risk.",
            )
            new_alerts.append(alert)
        
        # Store alerts
        self.alerts.extend(new_alerts)
        
        return matched_id, new_alerts
    
    def _match_fingerprint(self, fingerprint: torch.Tensor) -> Optional[int]:
        """Match a fingerprint against active campaigns using cosine similarity."""
        best_match = None
        best_similarity = 0.0
        
        for cid, campaign in self.campaigns.items():
            if campaign.status != "ACTIVE" or campaign.fingerprint is None:
                continue
            
            similarity = torch.nn.functional.cosine_similarity(
                fingerprint.unsqueeze(0),
                campaign.fingerprint.unsqueeze(0),
            ).item()
            
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match = cid
        
        return best_match
    
    def _create_campaign(
        self, incident_id: str, threat_score: float,
        fingerprint: Optional[torch.Tensor], region: Optional[str],
        timestamp: float,
    ) -> int:
        """Create a new campaign."""
        cid = self._next_campaign_id
        self._next_campaign_id += 1
        
        campaign = Campaign(
            campaign_id=cid,
            created_at=timestamp,
            updated_at=timestamp,
            incident_ids=[incident_id],
            incident_count=1,
            severity_scores=[threat_score],
            current_severity=threat_score,
            peak_severity=threat_score,
            fingerprint=fingerprint.clone() if fingerprint is not None else None,
            regions=[region] if region else [],
        )
        
        self.campaigns[cid] = campaign
        return cid
    
    def _expire_campaigns(self, now: float):
        """Mark campaigns as expired if dormant too long."""
        expiry_threshold = now - (self.expiry_hours * 3600)
        
        for cid, campaign in self.campaigns.items():
            if campaign.status == "ACTIVE" and campaign.updated_at < expiry_threshold:
                campaign.status = "EXPIRED"
    
    def _get_alert_severity(self, threat_score: float) -> str:
        """Map threat score to alert severity."""
        if threat_score >= self.alert_levels.get("critical", 0.9):
            return "CRITICAL"
        elif threat_score >= self.alert_levels.get("high", 0.7):
            return "HIGH"
        elif threat_score >= self.alert_levels.get("medium", 0.5):
            return "MEDIUM"
        return "LOW"
    
    def _create_alert(self, **kwargs) -> Alert:
        """Create and register an alert."""
        self._next_alert_id += 1
        alert = Alert(
            alert_id=f"ALERT-{self._next_alert_id:04d}",
            timestamp=time.time(),
            **kwargs,
        )
        return alert
    
    def get_active_campaigns(self) -> List[Dict]:
        """Get all active campaigns."""
        return [
            c.to_dict()
            for c in self.campaigns.values()
            if c.status == "ACTIVE"
        ]
    
    def get_campaign(self, campaign_id: int) -> Optional[Dict]:
        """Get a specific campaign."""
        campaign = self.campaigns.get(campaign_id)
        return campaign.to_dict() if campaign else None
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Get the most recent alerts."""
        return [a.to_dict() for a in sorted(self.alerts, key=lambda a: a.timestamp, reverse=True)[:limit]]
    
    def get_statistics(self) -> Dict:
        """Get campaign management statistics."""
        active = [c for c in self.campaigns.values() if c.status == "ACTIVE"]
        return {
            "total_campaigns": len(self.campaigns),
            "active_campaigns": len(active),
            "total_incidents": sum(c.incident_count for c in self.campaigns.values()),
            "total_alerts": len(self.alerts),
            "avg_severity": round(
                np.mean([c.avg_severity for c in active]) if active else 0.0, 4
            ),
            "regions_affected": list(set(
                r for c in active for r in c.regions
            )),
        }
