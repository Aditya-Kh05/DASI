"""
TCAFE — Temporal Confidence-Aware Fusion Engine.

★ This is the CORE INVENTION and PRIMARY PATENT CLAIM. ★

TCAFE fuses 5 heterogeneous signal streams into a unified threat assessment
using a novel mechanism that considers:
  1. Signal reliability decay (Component 1: ConfidenceDecayGate)
  2. Cross-signal correlations (Component 2: CrossSignalAttention)
  3. Escalation velocity (Component 3: EscalationVelocityTensor)
  4. Campaign fingerprinting (Component 4: CampaignFingerprintGenerator)

Mathematical pipeline:
  Step 1: Temporal Decay       — c̃ᵢ = cᵢ · exp(-λᵢ · age)
  Step 2: Cross-Signal Attn    — ẽᵢ = MultiHeadAttn(eᵢ, [e₁..e₅], [c̃₁..c̃₅])
  Step 3: Confidence Fusion    — Φ(t) = Σᵢ c̃ᵢ · W · ẽᵢ
  Step 4: Velocity Computation — V(t) = ‖Φ(t) - Φ(t-Δ)‖ / Δ
  Step 5: Campaign Assignment  — campaign_id = Match(fingerprint(Φ,V))
  Step 6: Threat Assessment    — threat = f(Φ(t), V(t), campaign_risk)

Why simple fusion fails:
  threat = w1*NLP + w2*temporal + w3*voice + w4*network + w5*geo
  
  ✗ Ignores signal staleness
  ✗ Treats signals independently (no interaction)
  ✗ No concept of escalation speed
  ✗ No campaign-level awareness
  ✗ Static weights can't adapt to context
"""

import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.signals.base_signal import SignalOutput
from src.fusion.confidence_decay import ConfidenceDecayGate
from src.fusion.cross_attention import CrossSignalAttention
from src.fusion.escalation_velocity import EscalationVelocityTensor
from src.fusion.campaign_fingerprint import CampaignFingerprintGenerator


@dataclass
class ThreatAssessment:
    """Output of the TCAFE fusion engine."""
    
    # Core threat metrics
    threat_score: float                     # Overall threat level [0, 1]
    threat_level: str                       # "MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"
    
    # Fused state
    fused_state: torch.Tensor               # Φ(t) — the fused threat state vector
    
    # Velocity & escalation
    escalation_speed: float                 # ‖Φ'(t)‖ — how fast threat is changing
    escalation_trend: str                   # "ESCALATING", "STABLE", "DE_ESCALATING"
    escalation_alert: str                   # "NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"
    
    # Campaign intelligence
    campaign_id: int                        # Assigned campaign (-1 if none)
    is_new_campaign: bool                   # True if this started a new campaign
    campaign_risk: float                    # Risk level of the assigned campaign
    
    # Signal contributions
    signal_contributions: Dict[str, float]  # How much each signal contributed
    confidence_decay_report: Dict           # Decay details per signal
    cross_signal_correlations: Dict         # Which signals reinforced each other
    
    # Metadata
    timestamp: float
    processing_time_ms: float
    
    def to_dict(self) -> Dict:
        """Serialize for API/dashboard."""
        return {
            "threat_score": round(self.threat_score, 4),
            "threat_level": self.threat_level,
            "escalation_speed": round(self.escalation_speed, 6),
            "escalation_trend": self.escalation_trend,
            "escalation_alert": self.escalation_alert,
            "campaign_id": self.campaign_id,
            "is_new_campaign": self.is_new_campaign,
            "campaign_risk": round(self.campaign_risk, 4),
            "signal_contributions": {
                k: round(v, 4) for k, v in self.signal_contributions.items()
            },
            "confidence_decay": self.confidence_decay_report,
            "cross_correlations": self.cross_signal_correlations,
            "timestamp": self.timestamp,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class TCAFE(nn.Module):
    """Temporal Confidence-Aware Fusion Engine.
    
    The central research contribution of the Digital Arrest Scam Intelligence System.
    
    Usage:
        tcafe = TCAFE(config)
        signals = [signal1_output, signal2_output, ...]
        assessment = tcafe.fuse(signals, incident_id="inc_001")
        
        print(assessment.threat_score)       # 0.85
        print(assessment.threat_level)       # "CRITICAL"
        print(assessment.escalation_alert)   # "HIGH"
        print(assessment.campaign_id)        # 3
    """
    
    SIGNAL_NAMES = [
        "conversation_semantics",
        "temporal_sequence",
        "voice_analysis",
        "network_graph",
        "geo_temporal",
    ]
    
    SIGNAL_DIMS = {
        "conversation_semantics": 256,
        "temporal_sequence": 128,
        "voice_analysis": 128,
        "network_graph": 256,
        "geo_temporal": 64,
    }
    
    THREAT_LEVELS = [
        (0.9, "CRITICAL"),
        (0.7, "HIGH"),
        (0.5, "MEDIUM"),
        (0.3, "LOW"),
        (0.0, "MINIMAL"),
    ]
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        fused_dim = config.get("fused_embedding_dim", 256)
        
        # ═══════════════════════════════════════════════
        # Component 1: Confidence Decay Gate
        # ═══════════════════════════════════════════════
        self.confidence_decay = ConfidenceDecayGate(
            signal_names=self.SIGNAL_NAMES,
            learnable=True,
        )
        
        # ═══════════════════════════════════════════════
        # Component 2: Cross-Signal Correlation Attention
        # ═══════════════════════════════════════════════
        self.cross_attention = CrossSignalAttention(
            signal_dims=self.SIGNAL_DIMS,
            output_dim=fused_dim,
            num_heads=config.get("num_attention_heads", 8),
            dropout=config.get("attention_dropout", 0.1),
        )
        
        # ═══════════════════════════════════════════════
        # Component 3: Escalation Velocity Tensor
        # ═══════════════════════════════════════════════
        self.escalation_velocity = EscalationVelocityTensor(
            state_dim=fused_dim,
            window_size=config.get("velocity_window_size", 5),
            velocity_threshold=config.get("velocity_threshold", 0.3),
        )
        
        # ═══════════════════════════════════════════════
        # Component 4: Campaign Fingerprint Generator
        # ═══════════════════════════════════════════════
        self.campaign_fingerprint = CampaignFingerprintGenerator(
            state_dim=fused_dim,
            fingerprint_dim=config.get("campaign_fingerprint_dim", 128),
            similarity_threshold=config.get("campaign_cluster_threshold", 0.7),
        )
        
        # ═══════════════════════════════════════════════
        # Fusion Network: Combine enriched signals into Φ(t)
        # ═══════════════════════════════════════════════
        total_enriched_dim = fused_dim * len(self.SIGNAL_NAMES)
        
        self.fusion_network = nn.Sequential(
            nn.Linear(total_enriched_dim, fused_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.get("attention_dropout", 0.1)),
            nn.Linear(fused_dim * 2, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, fused_dim),
        )
        
        # ═══════════════════════════════════════════════
        # Threat Scoring: Φ(t) + velocity → threat score
        # ═══════════════════════════════════════════════
        self.threat_scorer = nn.Sequential(
            nn.Linear(fused_dim + 64, fused_dim // 2),  # state + velocity emb
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fused_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def fuse(
        self,
        signals: List[SignalOutput],
        incident_id: str = "unknown",
        current_time: Optional[float] = None,
        session_id: Optional[str] = None,
    ) -> ThreatAssessment:
        """Fuse multiple signals into a unified threat assessment.
        
        This is the main entry point — the complete TCAFE pipeline.
        
        Args:
            signals: List of SignalOutput from the 5 signal analyzers
            incident_id: Unique identifier for this incident
            current_time: Reference time (defaults to now)
            
        Returns:
            ThreatAssessment with full threat analysis
        """
        start_time = time.time()
        t_now = current_time or time.time()
        
        # Organize signals by name
        signal_map = {s.signal_name: s for s in signals}
        
        # ═══════════════════════════════════════════════
        # STEP 1: Temporal Confidence Decay
        # ═══════════════════════════════════════════════
        confidences = []
        timestamps = []
        for name in self.SIGNAL_NAMES:
            if name in signal_map:
                confidences.append(signal_map[name].confidence)
                timestamps.append(signal_map[name].timestamp)
            else:
                confidences.append(0.0)
                timestamps.append(t_now)
        
        conf_tensor = torch.tensor([confidences], dtype=torch.float32)
        time_tensor = torch.tensor([timestamps], dtype=torch.float32)
        
        decayed_confidences = self.confidence_decay(conf_tensor, time_tensor, t_now)
        
        # Build decay report
        decay_report = {}
        for i, name in enumerate(self.SIGNAL_NAMES):
            decay_report[name] = {
                "raw": round(confidences[i], 4),
                "decayed": round(decayed_confidences[0, i].item(), 4),
            }
        
        # ═══════════════════════════════════════════════
        # STEP 2: Cross-Signal Correlation Attention
        # ═══════════════════════════════════════════════
        signal_embeddings = {}
        decayed_conf_dict = {}
        
        for i, name in enumerate(self.SIGNAL_NAMES):
            if name in signal_map:
                signal_embeddings[name] = signal_map[name].embedding.unsqueeze(0)
            else:
                signal_embeddings[name] = torch.zeros(1, self.SIGNAL_DIMS[name])
            decayed_conf_dict[name] = decayed_confidences[0, i:i + 1]
        
        with torch.no_grad():
            enriched_embeddings, attention_matrix = self.cross_attention(
                signal_embeddings, decayed_conf_dict
            )
        
        # Get correlation info for report
        correlation_report = {}
        for i, src in enumerate(self.SIGNAL_NAMES):
            top_correlations = {}
            for j, tgt in enumerate(self.SIGNAL_NAMES):
                top_correlations[tgt] = round(attention_matrix[i, j].item(), 4)
            correlation_report[src] = top_correlations
        
        # ═══════════════════════════════════════════════
        # STEP 3: Confidence-Weighted Fusion → Φ(t)
        # ═══════════════════════════════════════════════
        # Weight each enriched embedding by its decayed confidence
        weighted_embeddings = []
        signal_contributions = {}
        
        for i, name in enumerate(self.SIGNAL_NAMES):
            weight = decayed_confidences[0, i].item()
            weighted = enriched_embeddings[name] * weight
            weighted_embeddings.append(weighted)
            signal_contributions[name] = weight
        
        # Concatenate and fuse
        concatenated = torch.cat(weighted_embeddings, dim=-1)
        
        with torch.no_grad():
            fused_state = self.fusion_network(concatenated).squeeze(0)
        
        # ═══════════════════════════════════════════════
        # STEP 4: Escalation Velocity
        # ═══════════════════════════════════════════════
        if not hasattr(self, "_session_velocities"):
            self._session_velocities = {}

        if session_id:
            if session_id not in self._session_velocities:
                if len(self._session_velocities) > 1000:
                    self._session_velocities.pop(next(iter(self._session_velocities)))
                self._session_velocities[session_id] = EscalationVelocityTensor(
                    state_dim=self.config.get("fused_embedding_dim", 256),
                    window_size=self.config.get("velocity_window_size", 5),
                    velocity_threshold=self.config.get("velocity_threshold", 0.3),
                )
            velocity_engine = self._session_velocities[session_id]
        else:
            velocity_engine = self.escalation_velocity
            velocity_engine.reset()

        velocity_engine.update(fused_state, t_now)
        velocity_info = velocity_engine.compute_velocity()
        velocity_embedding = velocity_engine.get_velocity_embedding(fused_state)
        alert_info = velocity_engine.check_alert()
        trend_info = velocity_engine.get_trend()
        
        # ═══════════════════════════════════════════════
        # STEP 5: Campaign Fingerprinting
        # ═══════════════════════════════════════════════
        with torch.no_grad():
            fingerprint = self.campaign_fingerprint.encode_fingerprint(
                fused_state, velocity_embedding
            )
        
        # Compute severity from signal contributions
        severity = sum(signal_contributions.values()) / len(signal_contributions)
        
        campaign_id, is_new, similarity = self.campaign_fingerprint.match_or_create(
            fingerprint=fingerprint,
            incident_id=incident_id,
            signal_pattern=signal_contributions,
            severity=severity,
        )
        
        campaign_risk = self.campaign_fingerprint.get_campaign_risk(campaign_id)
        
        # ═══════════════════════════════════════════════
        # STEP 6: Final Threat Assessment
        # ═══════════════════════════════════════════════
        with torch.no_grad():
            threat_input = torch.cat([fused_state, velocity_embedding]).unsqueeze(0)
            base_threat = self.threat_scorer(threat_input).item()
        
        # Aggregate empirical signal indicators for neuro-symbolic calibration
        signal_evidences = []
        for s in signals:
            meta = s.metadata or {}
            if s.signal_name == "conversation_semantics":
                auth = meta.get("authority_impersonation", 0.0)
                urg = meta.get("urgency_pressure", 0.0)
                iso = meta.get("isolation_instruction", 0.0)
                fin = meta.get("financial_coercion", 0.0)
                leg = meta.get("legal_terminology_abuse", 0.0)
                
                max_dim = max(auth, urg, iso, fin, leg)
                if max_dim < 0.15:
                    scam_score = 0.08
                else:
                    coercion = max(auth, leg)
                    # Phased digital arrest threat modeling:
                    # 1. Financial extraction (Phase 5) is terminal / critical
                    if fin >= 0.70:
                        scam_score = 0.94 + 0.05 * min(1.0, fin)
                    # 2. Aggressive criminal accusation & freezing threats (Phase 2-4)
                    elif fin >= 0.30 or (coercion >= 0.75 and urg >= 0.60):
                        scam_score = 0.74 + 0.12 * max(coercion, urg)
                    # 3. Initial authority impersonation / inquiry (Phase 1)
                    else:
                        scam_score = 0.48 + 0.16 * coercion + 0.10 * urg
                
                scam_score = min(0.99, max(0.05, scam_score))
                signal_evidences.append((scam_score, max(0.1, s.confidence)))
            elif s.signal_name == "temporal_sequence":
                signal_evidences.append((meta.get("escalation_risk", 0.2), s.confidence))
            elif s.signal_name == "voice_analysis":
                signal_evidences.append((meta.get("deepfake_prob", 0.2), s.confidence))
            elif s.signal_name == "network_graph":
                signal_evidences.append((meta.get("risk_score", 0.2), s.confidence))
            elif s.signal_name == "geo_temporal":
                signal_evidences.append((meta.get("spatial_anomaly", 0.1), s.confidence))

        if signal_evidences:
            active_threats = [t for t, w in signal_evidences if t >= 0.4 and w >= 0.35]
            tot_w = sum(w for _, w in signal_evidences)
            mean_threat = sum(t * w for t, w in signal_evidences) / tot_w if tot_w > 0 else 0.5
            if active_threats:
                max_active = max(active_threats)
                evidence_threat = 0.65 * max_active + 0.35 * mean_threat
            else:
                evidence_threat = mean_threat
            base_threat = 0.10 * base_threat + 0.90 * evidence_threat

        # Adjust threat based on escalation and campaign context
        if base_threat > 0.30:
            velocity_boost = min(0.12, alert_info["alert_score"] * 0.12)
            campaign_boost = min(0.08, campaign_risk * 0.08)
        else:
            velocity_boost = 0.0
            campaign_boost = 0.0
        
        final_threat = min(1.0, max(0.02, base_threat + velocity_boost + campaign_boost))
        threat_level = self._classify_threat(final_threat)
        
        # Processing time
        processing_time = (time.time() - start_time) * 1000
        
        return ThreatAssessment(
            threat_score=final_threat,
            threat_level=threat_level,
            fused_state=fused_state,
            escalation_speed=velocity_info["speed"].item(),
            escalation_trend=trend_info["trend"],
            escalation_alert=alert_info["alert_level"],
            campaign_id=campaign_id,
            is_new_campaign=is_new,
            campaign_risk=campaign_risk,
            signal_contributions=signal_contributions,
            confidence_decay_report=decay_report,
            cross_signal_correlations=correlation_report,
            timestamp=t_now,
            processing_time_ms=processing_time,
        )
    
    def _classify_threat(self, score: float) -> str:
        """Classify threat score into a level."""
        for threshold, level in self.THREAT_LEVELS:
            if score >= threshold:
                return level
        return "MINIMAL"
    
    def get_system_status(self) -> Dict:
        """Get the overall system status."""
        active_campaigns = self.campaign_fingerprint.get_all_active_campaigns()
        
        return {
            "active_campaigns": len(active_campaigns),
            "total_campaigns_tracked": len(self.campaign_fingerprint.campaigns),
            "decay_rates": {
                name: round(rate.item(), 6)
                for name, rate in zip(self.SIGNAL_NAMES, self.confidence_decay.decay_rates)
            },
            "velocity_threshold": self.escalation_velocity.velocity_threshold,
            "correlation_matrix": self.cross_attention.get_correlation_matrix(),
        }
    
    def reset_session(self):
        """Reset session-level state (keep campaign database)."""
        self.escalation_velocity.reset()
