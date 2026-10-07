"""
Unit tests for all 5 signal analyzers in DASI.
Tests BaseSignalAnalyzer contract, embedding output shape, and metadata.
"""

import pytest
import torch
import numpy as np
import yaml

from src.signals.base_signal import SignalOutput, BaseSignalAnalyzer
from src.signals.conversation_semantics import ConversationSemanticsAnalyzer
from src.signals.temporal_sequence import TemporalSequenceAnalyzer
from src.signals.voice_analysis import VoiceAnalysisAnalyzer
from src.signals.network_graph import NetworkGraphAnalyzer
from src.signals.geo_temporal import GeoTemporalAnalyzer


@pytest.fixture(scope="module")
def model_config():
    with open("config/model_config.yaml", "r") as f:
        return yaml.safe_load(f)


class TestBaseSignal:
    def test_signal_output_structure(self):
        emb = torch.randn(1, 128)
        output = SignalOutput(
            signal_name="test_signal",
            embedding=emb,
            confidence=0.85,
            timestamp=100.0,
            metadata={"key": "val"}
        )
        assert output.signal_name == "test_signal"
        assert output.confidence == 0.85
        assert output.timestamp == 100.0
        assert output.metadata["key"] == "val"
        d = output.to_dict()
        assert d["signal_name"] == "test_signal"
        assert d["confidence"] == 0.85


class TestConversationSemantics:
    def test_analysis_output(self, model_config):
        cfg = model_config["conversation_semantics"]
        analyzer = ConversationSemanticsAnalyzer(cfg, device="cpu")
        
        text = "This is DCP Cyber Crime Delhi Police. Your Aadhaar is involved in terror funding."
        output = analyzer.analyze(text)
        
        assert isinstance(output, SignalOutput)
        assert output.signal_name == "conversation_semantics"
        assert output.embedding.shape[-1] == cfg["embedding_dim"]
        assert 0.0 <= output.confidence <= 1.0
        assert "authority_impersonation" in output.metadata
        assert "financial_coercion" in output.metadata


class TestTemporalSequence:
    def test_temporal_analysis(self, model_config):
        cfg = model_config["temporal_sequence"]
        analyzer = TemporalSequenceAnalyzer(cfg, device="cpu")
        
        # Test input with messages indicating sequence of phases
        data = {
            "messages": [
                {"text": "Hello, I am calling from telecom department.", "timestamp": 0.0},
                {"text": "A case has been filed against you under PMLA.", "timestamp": 60.0},
                {"text": "Police is coming with non-bailable arrest warrant.", "timestamp": 120.0},
            ]
        }
        output = analyzer.analyze(data)
        
        assert isinstance(output, SignalOutput)
        assert output.signal_name == "temporal_sequence"
        assert output.embedding.shape[-1] == cfg["embedding_dim"]
        assert 0.0 <= output.confidence <= 1.0


class TestVoiceAnalysis:
    def test_synthetic_voice_feature_analysis(self, model_config):
        cfg = model_config["voice_analysis"]
        analyzer = VoiceAnalysisAnalyzer(cfg, device="cpu")
        
        # Pass synthetic audio features or placeholder input
        data = {
            "audio_features": np.random.randn(cfg.get("n_mfcc", 40), 100).astype(np.float32),
            "sample_rate": 16000
        }
        output = analyzer.analyze(data)
        
        assert isinstance(output, SignalOutput)
        assert output.signal_name == "voice_analysis"
        assert output.embedding.shape[-1] == cfg["embedding_dim"]
        assert 0.0 <= output.confidence <= 1.0


class TestNetworkGraph:
    def test_network_analysis(self, model_config):
        cfg = model_config["network_graph"]
        analyzer = NetworkGraphAnalyzer(cfg, device="cpu")
        
        data = {
            "caller_id": "+919876543210",
            "callee_id": "+919123456789",
            "events": [
                {"src": "+919876543210", "dst": "+919123456789", "type": "call"}
            ]
        }
        output = analyzer.analyze(data)
        
        assert isinstance(output, SignalOutput)
        assert output.signal_name == "network_graph"
        assert output.embedding.shape[-1] == cfg["embedding_dim"]
        assert 0.0 <= output.confidence <= 1.0


class TestGeoTemporal:
    def test_geo_analysis(self, model_config):
        cfg = model_config["geo_temporal"]
        analyzer = GeoTemporalAnalyzer(cfg, device="cpu")
        
        data = {
            "latitude": 28.6139,
            "longitude": 77.2090,
            "timestamp": 1700000000.0,
            "region": "Delhi"
        }
        output = analyzer.analyze(data)
        
        assert isinstance(output, SignalOutput)
        assert output.signal_name == "geo_temporal"
        assert output.embedding.shape[-1] == cfg["embedding_dim"]
        assert 0.0 <= output.confidence <= 1.0
