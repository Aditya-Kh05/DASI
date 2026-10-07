"""End-to-end pipeline test."""
import json
from src.pipeline.orchestrator import DASIOrchestrator

# Initialize from config
orchestrator = DASIOrchestrator.from_config()

# Test with a scam scenario
result = orchestrator.process({
    "messages": [
        {"role": "caller", "text": "This is Inspector Verma from CBI. Your Aadhaar has been used in money laundering.", "timestamp_offset": 0},
        {"role": "victim", "text": "What? That is impossible!", "timestamp_offset": 5},
        {"role": "caller", "text": "You are under digital arrest. Do not disconnect this call or you will be arrested.", "timestamp_offset": 12},
        {"role": "caller", "text": "Transfer 5 lakh rupees immediately to the verification account.", "timestamp_offset": 25},
    ],
    "language": "en",
    "region": "Delhi NCR",
    "caller_id": "+91-9876543210",
})

print("=== PIPELINE RESULT ===")
print("Incident:", result.incident_id)
print("Threat Score:", round(result.threat_assessment.threat_score, 4))
print("Threat Level:", result.threat_assessment.threat_level)
print("Escalation:", result.threat_assessment.escalation_trend)
print("Campaign: C-%03d" % result.campaign_id)
print("Alerts:", len(result.campaign_alerts))
print("Processing Time: %.1fms" % result.processing_time_ms)
print("Signals Available:", result.signals_available)
print("Signals Missing:", result.signals_missing)
print()
print("Signal Contributions:")
for sig, weight in result.threat_assessment.signal_contributions.items():
    print("  %s: %.4f" % (sig, weight))
print()

# Process a second incident to test campaign matching
result2 = orchestrator.process({
    "messages": [
        {"role": "caller", "text": "This is officer from enforcement directorate. You are implicated in a terror funding case.", "timestamp_offset": 0},
        {"role": "victim", "text": "Please, I am innocent!", "timestamp_offset": 8},
    ],
    "language": "en",
    "region": "Mumbai Metropolitan",
})

print("=== SECOND INCIDENT ===")
print("Threat Score:", round(result2.threat_assessment.threat_score, 4))
print("Campaign: C-%03d" % result2.campaign_id)
print()

status = orchestrator.get_system_status()
print("System Status:")
print("  Total Processed:", status["total_processed"])
print("  Scams Detected:", status["total_scams_detected"])
print("  Active Campaigns:", status["active_campaigns"])
print()
print("=== FULL PIPELINE OPERATIONAL ===")
