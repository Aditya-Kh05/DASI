"""
Unit and integration tests for the DASI Pipeline Orchestrator and InputProcessor.
"""

import pytest
import yaml

from src.pipeline.input_processor import InputProcessor, IncidentInput
from src.pipeline.orchestrator import DASIOrchestrator, PipelineResult


class TestInputProcessor:
    def test_process_valid_input(self):
        processor = InputProcessor()
        raw = {
            "messages": [
                {"role": "caller", "text": "This is police.", "timestamp_offset": 0}
            ],
            "language": "en",
            "region": "Delhi NCR",
            "caller_id": "+91-9876543210"
        }
        incident = processor.process(raw)
        assert isinstance(incident, IncidentInput)
        assert incident.incident_id.startswith("INC-")
        assert len(incident.messages) == 1
        assert incident.region == "Delhi NCR"

    def test_process_empty_messages(self):
        processor = InputProcessor()
        raw = {"language": "en"}
        with pytest.raises(ValueError, match="At least one message is required"):
            processor.process(raw)


class TestOrchestrator:
    @pytest.fixture(scope="module")
    def orchestrator(self):
        return DASIOrchestrator.from_config()

    def test_end_to_end_scam_detection(self, orchestrator):
        raw = {
            "messages": [
                {"role": "caller", "text": "This is Inspector Sharma from CBI headquarters.", "timestamp_offset": 0},
                {"role": "caller", "text": "Your Aadhaar is linked to illegal transactions. Non-bailable arrest warrant issued.", "timestamp_offset": 10},
                {"role": "caller", "text": "Transfer money to the government safe account now.", "timestamp_offset": 30},
            ],
            "language": "en",
            "region": "Delhi NCR",
            "caller_id": "+91-9876543210",
        }
        result = orchestrator.process(raw)
        
        assert isinstance(result, PipelineResult)
        assert result.incident_id is not None
        assert 0.0 <= result.threat_assessment.threat_score <= 1.0
        assert result.threat_assessment.threat_level in ["MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert result.campaign_id >= 0
        assert result.processing_time_ms > 0
        assert len(result.signals_available) + len(result.signals_missing) == 5

    def test_system_status(self, orchestrator):
        status = orchestrator.get_system_status()
        assert status["status"] == "OPERATIONAL"
        assert status["total_processed"] >= 1
        assert "detection_rate" in status
        assert "active_campaigns" in status
