"""
Unit tests for Campaign Intelligence Module:
  - CampaignManager (lifecycle, matching, merging)
  - AlertEngine (rule + ML alert generation)
  - CampaignStore (in-memory + JSON persistence)
"""

import os
import pytest
import torch
import tempfile

from src.campaign.campaign_manager import CampaignManager, Campaign, Alert
from src.campaign.alert_engine import AlertEngine
from src.campaign.campaign_store import CampaignStore


class TestCampaignManager:
    def test_campaign_lifecycle(self):
        manager = CampaignManager({
            "max_active_campaigns": 100,
            "campaign_expiry_hours": 168,
            "fingerprint_similarity_threshold": 0.7,
        })
        
        fp = torch.randn(128)
        # Incident 1 creates a new campaign
        cid1, alerts1 = manager.process_incident(
            incident_id="inc-101",
            threat_score=0.85,
            fingerprint=fp,
            region="Delhi",
            escalation_speed=0.2
        )
        assert cid1 >= 0
        assert manager.campaigns[cid1].incident_count == 1
        
        # Incident 2 with identical fingerprint joins existing campaign
        cid2, alerts2 = manager.process_incident(
            incident_id="inc-102",
            threat_score=0.90,
            fingerprint=fp,
            region="Mumbai",
            escalation_speed=0.4
        )
        assert cid2 == cid1
        assert manager.campaigns[cid1].incident_count == 2
        assert "Mumbai" in manager.campaigns[cid1].regions


class TestAlertEngine:
    def test_alert_generation(self):
        engine = AlertEngine({
            "early_warning_threshold": 0.4,
            "critical_threat_threshold": 0.85,
            "escalation_speed_threshold": 0.3,
        })
        
        # Test critical alert
        alerts = engine.evaluate(
            threat_score=0.95,
            threat_level="CRITICAL",
            escalation_speed=0.5,
            incident_id="inc-test-crit",
            campaign_id=1,
            region="Bengaluru",
            is_deepfake=True,
        )
        assert len(alerts) > 0
        severities = [a["severity"] for a in alerts]
        assert "CRITICAL" in severities or "HIGH" in severities
        
        alert_types = [a["alert_type"] for a in alerts]
        assert "DEEPFAKE_DETECTED" in alert_types or "CRITICAL_THREAT" in alert_types or "ESCALATION" in alert_types


class TestCampaignStore:
    def test_store_and_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name
        
        try:
            store = CampaignStore(persist_path=temp_path)
            store.store_campaign(1, {"campaign_id": 1, "severity": 0.85, "incidents": ["i1", "i2"]})
            store.store_alert({"alert_id": "ALT-001", "type": "EARLY_WARNING", "timestamp": 100.0})
            store.save()
            
            # Load into new store instance
            store2 = CampaignStore(persist_path=temp_path)
            store2.load()
            camp = store2.get_campaign(1)
            assert camp is not None
            assert camp["campaign_id"] == 1
            assert len(store2.get_alerts()) == 1
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
