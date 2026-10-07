"""
Unit tests for TCAFE Fusion Engine and its 4 core sub-components:
  1. ConfidenceDecayGate
  2. CrossSignalAttention
  3. EscalationVelocityTensor
  4. CampaignFingerprintGenerator
  5. Full TCAFE module
"""

import pytest
import torch
import yaml

from src.signals.base_signal import SignalOutput
from src.fusion.confidence_decay import ConfidenceDecayGate
from src.fusion.cross_attention import CrossSignalAttention
from src.fusion.escalation_velocity import EscalationVelocityTensor
from src.fusion.campaign_fingerprint import CampaignFingerprintGenerator
from src.fusion.tcafe import TCAFE, ThreatAssessment


@pytest.fixture(scope="module")
def model_config():
    with open("config/model_config.yaml", "r") as f:
        return yaml.safe_load(f)


class TestConfidenceDecayGate:
    def test_decay_computation(self):
        signals = ["conversation_semantics", "temporal_sequence", "voice_analysis", "network_graph", "geo_temporal"]
        gate = ConfidenceDecayGate(signal_names=signals)
        
        # Fresh signal (dt = 0) vs aged signal (dt = 100)
        confidences = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9]])
        now = 1000.0
        timestamps_fresh = torch.tensor([[now, now, now, now, now]])
        decayed_fresh = gate(confidences, timestamps_fresh, current_time=now)
        assert torch.allclose(decayed_fresh, confidences, atol=1e-3)
        
        timestamps_aged = torch.tensor([[now - 100.0, now - 100.0, now - 100.0, now - 100.0, now - 100.0]])
        decayed_aged = gate(confidences, timestamps_aged, current_time=now)
        assert torch.all(decayed_aged <= confidences)
        assert torch.all(decayed_aged >= gate.confidence_floor)


class TestCrossSignalAttention:
    def test_attention_shape_and_weights(self):
        signal_dims = {
            "conversation_semantics": 256,
            "temporal_sequence": 128,
            "voice_analysis": 128,
            "network_graph": 256,
            "geo_temporal": 64,
        }
        attn = CrossSignalAttention(signal_dims=signal_dims, output_dim=256, num_heads=8)
        
        embeddings = {
            name: torch.randn(1, dim) for name, dim in signal_dims.items()
        }
        weights = {name: 0.8 for name in signal_dims}
        
        enriched, attn_matrix = attn(embeddings, weights)
        assert len(enriched) == len(signal_dims)
        for name, emb in enriched.items():
            assert emb.shape == (1, 256)
        assert attn_matrix.shape == (len(signal_dims), len(signal_dims))


class TestEscalationVelocityTensor:
    def test_velocity_and_acceleration(self):
        tracker = EscalationVelocityTensor(state_dim=256, window_size=5, velocity_threshold=0.3)
        
        # Initial step
        s0 = torch.zeros(256)
        tracker.update(s0, timestamp=0.0)
        v0 = tracker.compute_velocity()
        assert v0["speed"].item() == 0.0
        
        # Second step with delta
        s1 = torch.ones(256) * 2.0
        tracker.update(s1, timestamp=1.0)
        v1 = tracker.compute_velocity()
        assert v1["speed"].item() > 0.0
        
        alert = tracker.check_alert()
        assert "alert_level" in alert
        assert "speed" in alert


class TestCampaignFingerprintGenerator:
    def test_fingerprint_generation(self):
        gen = CampaignFingerprintGenerator(state_dim=256, fingerprint_dim=128, similarity_threshold=0.7)
        
        state = torch.randn(256)
        velocity_embedding = torch.randn(64)
        fp = gen.encode_fingerprint(state, velocity_embedding)
        assert fp.shape[-1] == 128
        
        # Test matching / cluster assignment
        pattern = {"conversation_semantics": 0.8, "temporal_sequence": 0.7}
        cid1, is_new1, sim1 = gen.match_or_create(fp, "inc-1", pattern, severity=0.8)
        assert is_new1 is True
        
        # Match with same fingerprint should match cid1
        cid2, is_new2, sim2 = gen.match_or_create(fp, "inc-2", pattern, severity=0.85)
        assert cid2 == cid1
        assert is_new2 is False


class TestTCAFEEndToEnd:
    def test_full_fusion(self, model_config):
        cfg = model_config["tcafe"]
        tcafe = TCAFE(cfg)
        tcafe.eval()
        
        now = 1000.0
        signals = [
            SignalOutput(
                signal_name="conversation_semantics",
                embedding=torch.randn(256),
                confidence=0.85,
                timestamp=now - 10.0,
                metadata={"authority_impersonation": 0.9}
            ),
            SignalOutput(
                signal_name="temporal_sequence",
                embedding=torch.randn(128),
                confidence=0.75,
                timestamp=now - 5.0,
                metadata={"predicted_phase": 2}
            ),
            SignalOutput(
                signal_name="voice_analysis",
                embedding=torch.randn(128),
                confidence=0.60,
                timestamp=now - 2.0,
                metadata={"deepfake_prob": 0.3}
            ),
            SignalOutput(
                signal_name="network_graph",
                embedding=torch.randn(256),
                confidence=0.80,
                timestamp=now,
                metadata={"caller_node_risk": 0.7}
            ),
            SignalOutput(
                signal_name="geo_temporal",
                embedding=torch.randn(64),
                confidence=0.70,
                timestamp=now,
                metadata={"is_cluster": True}
            ),
        ]
        
        assessment = tcafe.fuse(signals, incident_id="test_incident", current_time=now)
        
        assert isinstance(assessment, ThreatAssessment)
        assert 0.0 <= assessment.threat_score <= 1.0
        assert assessment.threat_level in ["MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert assessment.fused_state.shape[-1] == cfg.get("fused_embedding_dim", 256)
        assert assessment.escalation_alert in ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert len(assessment.signal_contributions) == 5
