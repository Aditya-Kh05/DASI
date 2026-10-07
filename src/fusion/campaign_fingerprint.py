"""
Campaign Fingerprint Generator — TCAFE Component 4.

Creates unique fingerprints for scam campaigns by clustering fused threat
state trajectories. This enables:
  1. Matching new incidents to known campaigns
  2. Detecting novel campaigns (no fingerprint match)
  3. Tracking campaign evolution over time

Mathematical formulation:
  campaign_id = TemporalGraphCluster(Φ(t), historical_Φs)
  fingerprint = mean(Φ_trajectory) ⊕ velocity_signature ⊕ signal_pattern
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class CampaignFingerprint:
    """A unique fingerprint for a scam campaign."""
    campaign_id: int
    centroid: np.ndarray                   # Mean fused state
    velocity_signature: np.ndarray         # Characteristic velocity pattern
    signal_pattern: Dict[str, float]       # Which signals are dominant
    first_seen: float
    last_seen: float
    num_incidents: int = 0
    severity: float = 0.0
    active: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "campaign_id": self.campaign_id,
            "centroid_norm": float(np.linalg.norm(self.centroid)),
            "velocity_norm": float(np.linalg.norm(self.velocity_signature)),
            "signal_pattern": self.signal_pattern,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "num_incidents": self.num_incidents,
            "severity": round(self.severity, 4),
            "active": self.active,
            "duration_hours": round((self.last_seen - self.first_seen) / 3600, 2),
        }


class CampaignFingerprintGenerator(nn.Module):
    """Generate and match campaign fingerprints.
    
    Maintains a database of known campaign fingerprints and matches
    new incidents against them. Novel incidents that don't match
    any existing campaign may indicate a new campaign.
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        fingerprint_dim: int = 128,
        similarity_threshold: float = 0.75,
        max_campaigns: int = 1000,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.fingerprint_dim = fingerprint_dim
        self.similarity_threshold = similarity_threshold
        self.max_campaigns = max_campaigns
        
        # Fingerprint encoder
        self.fingerprint_encoder = nn.Sequential(
            nn.Linear(state_dim + 64, fingerprint_dim * 2),  # state + velocity
            nn.ReLU(),
            nn.Linear(fingerprint_dim * 2, fingerprint_dim),
            nn.ReLU(),
            nn.Linear(fingerprint_dim, fingerprint_dim),
        )
        
        # Campaign database
        self.campaigns: Dict[int, CampaignFingerprint] = {}
        self._next_campaign_id = 0
        
        # Incident-to-campaign mapping
        self._incident_assignments: Dict[str, int] = {}
    
    def encode_fingerprint(
        self,
        fused_state: torch.Tensor,
        velocity_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a state + velocity into a fingerprint vector.
        
        Args:
            fused_state: (state_dim,) — current fused threat state
            velocity_embedding: (64,) — velocity information
            
        Returns:
            (fingerprint_dim,) — campaign fingerprint vector
        """
        combined = torch.cat([fused_state, velocity_embedding])
        fingerprint = self.fingerprint_encoder(combined.unsqueeze(0)).squeeze(0)
        return fingerprint
    
    def match_or_create(
        self,
        fingerprint: torch.Tensor,
        incident_id: str,
        signal_pattern: Dict[str, float],
        severity: float = 0.0,
    ) -> Tuple[int, bool, float]:
        """Match an incident to an existing campaign or create a new one.
        
        Args:
            fingerprint: (fingerprint_dim,) — incident fingerprint
            incident_id: Unique identifier for this incident
            signal_pattern: Dict of signal_name → contribution score
            severity: Overall severity score
            
        Returns:
            Tuple of:
              - campaign_id: int
              - is_new: bool — True if a new campaign was created
              - similarity: float — match similarity (1.0 for new campaigns)
        """
        fp_numpy = fingerprint.detach().cpu().numpy()
        current_time = time.time()
        
        # Find best matching existing campaign
        best_match_id = -1
        best_similarity = 0.0
        
        for cid, campaign in self.campaigns.items():
            if not campaign.active:
                continue
            
            similarity = self._cosine_similarity(fp_numpy, campaign.centroid)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = cid
        
        # Match to existing or create new
        if best_similarity >= self.similarity_threshold and best_match_id >= 0:
            # Update existing campaign
            campaign = self.campaigns[best_match_id]
            campaign.num_incidents += 1
            campaign.last_seen = current_time
            campaign.severity = max(campaign.severity, severity)
            
            # Update centroid (running average)
            alpha = 1.0 / campaign.num_incidents
            campaign.centroid = (1 - alpha) * campaign.centroid + alpha * fp_numpy
            
            # Update signal pattern
            for sig_name, score in signal_pattern.items():
                old = campaign.signal_pattern.get(sig_name, 0.0)
                campaign.signal_pattern[sig_name] = (1 - alpha) * old + alpha * score
            
            self._incident_assignments[incident_id] = best_match_id
            return best_match_id, False, best_similarity
        
        else:
            # Create new campaign
            new_id = self._next_campaign_id
            self._next_campaign_id += 1
            
            self.campaigns[new_id] = CampaignFingerprint(
                campaign_id=new_id,
                centroid=fp_numpy,
                velocity_signature=np.zeros(64),
                signal_pattern=signal_pattern,
                first_seen=current_time,
                last_seen=current_time,
                num_incidents=1,
                severity=severity,
                active=True,
            )
            
            self._incident_assignments[incident_id] = new_id
            
            # Cleanup old campaigns if too many
            if len(self.campaigns) > self.max_campaigns:
                self._cleanup_old_campaigns()
            
            return new_id, True, 1.0
    
    def get_campaign_info(self, campaign_id: int) -> Optional[Dict]:
        """Get information about a specific campaign."""
        if campaign_id in self.campaigns:
            return self.campaigns[campaign_id].to_dict()
        return None
    
    def get_all_active_campaigns(self) -> List[Dict]:
        """Get all currently active campaigns."""
        return [
            c.to_dict() for c in self.campaigns.values()
            if c.active
        ]
    
    def get_campaign_risk(self, campaign_id: int) -> float:
        """Get the risk level of a campaign."""
        if campaign_id in self.campaigns:
            campaign = self.campaigns[campaign_id]
            # Risk based on: severity, number of incidents, duration
            size_factor = min(1.0, campaign.num_incidents / 10.0)
            return 0.5 * campaign.severity + 0.3 * size_factor + 0.2 * (1.0 if campaign.active else 0.0)
        return 0.0
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _cleanup_old_campaigns(self):
        """Remove oldest inactive campaigns when over capacity."""
        current_time = time.time()
        expiry = 7 * 24 * 3600  # 7 days
        
        to_remove = []
        for cid, campaign in self.campaigns.items():
            if current_time - campaign.last_seen > expiry:
                campaign.active = False
                to_remove.append(cid)
        
        # Keep the most recent ones
        if len(self.campaigns) - len(to_remove) > self.max_campaigns * 0.8:
            inactive = sorted(
                [(cid, c.last_seen) for cid, c in self.campaigns.items() if not c.active],
                key=lambda x: x[1],
            )
            to_remove.extend([cid for cid, _ in inactive[:len(inactive) // 2]])
        
        for cid in set(to_remove):
            del self.campaigns[cid]
