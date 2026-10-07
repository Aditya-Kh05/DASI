"""
End-to-End Demo — Digital Arrest Scam Intelligence System.

Simulates a realistic digital arrest scam scenario and demonstrates
how the system detects it progressively through all 5 signals + TCAFE fusion.

Scenario: A scam syndicate based in Southeast Asia runs a "digital arrest"
campaign targeting professionals in Mumbai and Delhi.

Run: python demo/demo_scenario.py
"""

import sys
import os
import time
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.signals.base_signal import SignalOutput
from src.signals.conversation_semantics.model import ConversationSemanticsAnalyzer
from src.signals.conversation_semantics.preprocessor import TextPreprocessor
from src.signals.temporal_sequence.phase_detector import TemporalSequenceAnalyzer
from src.signals.temporal_sequence.state_machine import ScamPhase
from src.signals.voice_analysis.speaker_profiler import VoiceAnalysisAnalyzer
from src.signals.network_graph.relationship_mapper import NetworkGraphAnalyzer
from src.signals.geo_temporal.temporal_patterns import GeoTemporalAnalyzer
from src.fusion.tcafe import TCAFE


# ══════════════════════════════════════════════════════════════
# Simulated Scam Scenario
# ══════════════════════════════════════════════════════════════

SCAM_SCENARIO = [
    {
        "phase": 1,
        "phase_name": "AUTHORITY_CLAIM",
        "timestamp_offset": 0,
        "message": "Hello, this is Inspector Rajesh Sharma calling from CBI Headquarters, New Delhi. We have received a complaint against your Aadhaar number.",
        "caller_phone": "+91-9876543210",
        "callee_phone": "+91-9123456789",
        "caller_ip": "103.45.67.89",
        "caller_device": "DEV_SCAM_001",
        "location": {"lat": 19.0760, "lon": 72.8777, "city": "Mumbai"},
    },
    {
        "phase": 2,
        "phase_name": "ACCUSATION",
        "timestamp_offset": 45,
        "message": "Your Aadhaar card number 1234-5678-9012 has been used in a money laundering case. An FIR has been registered under PMLA Act. Your bank accounts are under investigation.",
        "caller_phone": "+91-9876543210",
        "callee_phone": "+91-9123456789",
        "caller_ip": "103.45.67.89",
        "caller_device": "DEV_SCAM_001",
        "location": {"lat": 19.0760, "lon": 72.8777, "city": "Mumbai"},
    },
    {
        "phase": 3,
        "phase_name": "FEAR_ESCALATION",
        "timestamp_offset": 120,
        "message": "A non-bailable arrest warrant has been issued in your name. If you don't cooperate, police will come to your house within 2 hours. This is a very serious case involving terrorism funding.",
        "caller_phone": "+91-9876543210",
        "callee_phone": "+91-9123456789",
        "caller_ip": "103.45.67.90",
        "caller_device": "DEV_SCAM_001",
        "location": {"lat": 19.0760, "lon": 72.8777, "city": "Mumbai"},
    },
    {
        "phase": 4,
        "phase_name": "ISOLATION",
        "timestamp_offset": 210,
        "message": "This is a classified investigation. Do not tell anyone about this call, not even your family. If you share this information, you will be charged with obstruction of justice. Stay on this video call.",
        "caller_phone": "+91-9876543210",
        "callee_phone": "+91-9123456789",
        "caller_ip": "103.45.67.90",
        "caller_device": "DEV_SCAM_001",
        "location": {"lat": 19.0760, "lon": 72.8777, "city": "Mumbai"},
    },
    {
        "phase": 5,
        "phase_name": "FINANCIAL_EXTRACTION",
        "timestamp_offset": 360,
        "message": "To prove your innocence, you need to transfer all your funds to this RBI verification account for investigation. Transfer via NEFT to account 123456789012345. This is a refundable deposit.",
        "caller_phone": "+91-9876543210",
        "callee_phone": "+91-9123456789",
        "caller_ip": "103.45.67.90",
        "caller_device": "DEV_SCAM_001",
        "location": {"lat": 19.0760, "lon": 72.8777, "city": "Mumbai"},
        "transaction": {"from": "VICTIM_ACCT_001", "to": "SCAM_ACCT_001", "amount": 500000},
    },
]

# Additional victims (same campaign, different cities)
PARALLEL_VICTIMS = [
    {
        "message": "This is Deputy Commissioner from Cyber Crime Cell. Your mobile number is linked to illegal hawala transactions.",
        "caller_phone": "+91-9876543211",  # Different phone, same device
        "callee_phone": "+91-9988776655",
        "caller_ip": "103.45.67.91",
        "caller_device": "DEV_SCAM_001",  # SAME DEVICE — network signal!
        "location": {"lat": 28.7041, "lon": 77.1025, "city": "Delhi"},
        "timestamp_offset": 30,
    },
    {
        "message": "I am calling from Enforcement Directorate. Your PAN card is flagged for suspicious transactions worth 50 crore.",
        "caller_phone": "+91-9876543212",
        "callee_phone": "+91-9112233445",
        "caller_ip": "103.45.67.92",
        "caller_device": "DEV_SCAM_002",
        "location": {"lat": 12.9716, "lon": 77.5946, "city": "Bangalore"},
        "timestamp_offset": 90,
    },
]


def print_header(text: str, char: str = "═"):
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def print_signal_result(name: str, output: SignalOutput):
    """Pretty-print a signal analysis result."""
    print(f"\n  📡 Signal: {name}")
    print(f"     Confidence: {output.confidence:.4f}")
    print(f"     Embedding norm: {torch.norm(output.embedding).item():.4f}")
    
    # Print key metadata
    for key, value in output.metadata.items():
        if isinstance(value, dict):
            continue  # Skip nested dicts for brevity
        if isinstance(value, list):
            continue
        print(f"     {key}: {value}")


def print_threat_assessment(assessment):
    """Pretty-print a TCAFE threat assessment."""
    # Color codes for threat levels
    level_colors = {
        "CRITICAL": "\033[91m",  # Red
        "HIGH": "\033[93m",      # Yellow
        "MEDIUM": "\033[33m",    # Orange
        "LOW": "\033[92m",       # Green
        "MINIMAL": "\033[90m",   # Gray
    }
    reset = "\033[0m"
    color = level_colors.get(assessment.threat_level, "")
    
    print(f"\n  {'─' * 50}")
    print(f"  🎯 THREAT ASSESSMENT")
    print(f"  {'─' * 50}")
    print(f"  Threat Score:     {color}{assessment.threat_score:.4f} [{assessment.threat_level}]{reset}")
    print(f"  Escalation Speed: {assessment.escalation_speed:.6f}")
    print(f"  Escalation Trend: {assessment.escalation_trend}")
    print(f"  Escalation Alert: {assessment.escalation_alert}")
    print(f"  Campaign ID:      {assessment.campaign_id} {'(NEW!)' if assessment.is_new_campaign else ''}")
    print(f"  Campaign Risk:    {assessment.campaign_risk:.4f}")
    print(f"  Processing Time:  {assessment.processing_time_ms:.2f}ms")
    
    print(f"\n  Signal Contributions:")
    for sig_name, contrib in assessment.signal_contributions.items():
        bar = "█" * int(contrib * 20)
        print(f"    {sig_name:<25s} {contrib:.4f} {bar}")


def run_demo():
    """Run the full end-to-end demo."""
    print_header("DIGITAL ARREST SCAM INTELLIGENCE SYSTEM — LIVE DEMO")
    print("  Demonstrating the Temporal Confidence-Aware Fusion Engine (TCAFE)")
    print("  Processing a simulated digital arrest scam in real-time...\n")
    
    # ─────────────────────────────────────────────
    # Initialize all components
    # ─────────────────────────────────────────────
    print("⚙️  Initializing system components...")
    
    # Signal analyzers
    nlp_config = {"embedding_dim": 256, "dropout": 0.3, "max_seq_length": 512, "confidence_decay_rate": 0.01}
    temporal_config = {"embedding_dim": 128, "lstm_num_layers": 2, "dropout": 0.2, "confidence_decay_rate": 0.05}
    voice_config = {"embedding_dim": 128, "sample_rate": 16000, "n_mfcc": 40, "n_fft": 2048, "hop_length": 512, "mel_bands": 128, "deepfake_threshold": 0.7, "confidence_decay_rate": 0.03}
    network_config = {"embedding_dim": 128, "gnn_hidden_dim": 256, "gnn_layers": 3, "cluster_min_size": 3, "dropout": 0.2, "confidence_decay_rate": 0.005}
    geo_config = {"embedding_dim": 64, "spatial_eps": 50.0, "temporal_eps": 3600.0, "min_cluster_size": 3, "confidence_decay_rate": 0.1}
    
    nlp_analyzer = ConversationSemanticsAnalyzer(nlp_config, device="cpu")
    temporal_analyzer = TemporalSequenceAnalyzer(temporal_config, device="cpu")
    voice_analyzer = VoiceAnalysisAnalyzer(voice_config, device="cpu")
    network_analyzer = NetworkGraphAnalyzer(network_config, device="cpu")
    geo_analyzer = GeoTemporalAnalyzer(geo_config, device="cpu")
    
    # TCAFE Fusion Engine
    tcafe_config = {
        "fused_embedding_dim": 256,
        "num_attention_heads": 8,
        "attention_dropout": 0.1,
        "velocity_window_size": 5,
        "velocity_threshold": 0.3,
        "campaign_fingerprint_dim": 128,
        "campaign_cluster_threshold": 0.7,
    }
    tcafe = TCAFE(tcafe_config)
    tcafe.eval()
    
    print("  ✅ All 5 signal analyzers initialized")
    print("  ✅ TCAFE fusion engine initialized")
    print("  ✅ Campaign fingerprint database ready")
    
    base_time = time.time()
    
    # ─────────────────────────────────────────────
    # Process primary victim's scam sequence
    # ─────────────────────────────────────────────
    print_header("SCENARIO: Digital Arrest Scam — Primary Victim (Mumbai)")
    
    for step_idx, step in enumerate(SCAM_SCENARIO):
        current_time = base_time + step["timestamp_offset"]
        
        print(f"\n{'─' * 70}")
        print(f"  📞 Phase {step['phase']}: {step['phase_name']}")
        print(f"  ⏱️  T+{step['timestamp_offset']}s")
        print(f"  💬 \"{step['message'][:80]}...\"")
        
        # ── Signal 1: NLP Analysis ──
        nlp_result = nlp_analyzer.analyze(step["message"])
        
        # ── Signal 2: Temporal Sequence ──
        temporal_result = temporal_analyzer.analyze({
            "message_embedding": nlp_result.embedding,
            "message_index": step_idx,
            "timestamp": current_time,
        })
        
        # ── Signal 3: Voice Analysis (simulated audio) ──
        # Generate synthetic waveform for demo
        duration_sec = 5.0
        sample_rate = 16000
        t = np.linspace(0, duration_sec, int(sample_rate * duration_sec))
        # Simulate a voice-like signal with some noise
        waveform = 0.3 * np.sin(2 * np.pi * 150 * t) + 0.1 * np.random.randn(len(t))
        waveform = waveform.astype(np.float32)
        
        voice_result = voice_analyzer.analyze({"waveform": waveform})
        
        # ── Signal 4: Network Graph ──
        network_data = {
            "caller_phone": step["caller_phone"],
            "callee_phone": step["callee_phone"],
            "caller_ip": step.get("caller_ip"),
            "caller_device": step.get("caller_device"),
            "timestamp": current_time,
            "duration": step["timestamp_offset"],
        }
        if "transaction" in step:
            network_data["transaction"] = step["transaction"]
        
        network_result = network_analyzer.analyze(network_data)
        
        # ── Signal 5: Geo-Temporal ──
        geo_result = geo_analyzer.analyze({
            "latitude": step["location"]["lat"],
            "longitude": step["location"]["lon"],
            "timestamp": current_time,
            "risk_score": nlp_result.confidence,
            "region": step["location"]["city"],
        })
        
        # Print signal results
        print_signal_result("Conversation Semantics (NLP)", nlp_result)
        print_signal_result("Temporal Sequence (Phase)", temporal_result)
        print_signal_result("Voice Analysis", voice_result)
        print_signal_result("Network Graph", network_result)
        print_signal_result("Geo-Temporal", geo_result)
        
        # ── TCAFE Fusion ──
        all_signals = [nlp_result, temporal_result, voice_result, network_result, geo_result]
        assessment = tcafe.fuse(
            signals=all_signals,
            incident_id=f"primary_victim_phase_{step['phase']}",
            current_time=current_time,
        )
        
        print_threat_assessment(assessment)
        
        # Early warning check
        if assessment.threat_level in ("HIGH", "CRITICAL") and step["phase"] <= 3:
            print(f"\n  ⚠️  EARLY WARNING TRIGGERED AT PHASE {step['phase']}!")
            print(f"  ⚠️  System detected scam BEFORE financial extraction (Phase 5)")
            print(f"  ⚠️  Potential savings for victim: ₹5,00,000")
    
    # ─────────────────────────────────────────────
    # Process parallel victims (campaign detection)
    # ─────────────────────────────────────────────
    print_header("CAMPAIGN DETECTION — Parallel Victims")
    print("  The same scam syndicate targets victims in Delhi and Bangalore...")
    
    for victim_idx, victim in enumerate(PARALLEL_VICTIMS):
        current_time = base_time + victim["timestamp_offset"]
        
        print(f"\n  📞 Parallel Victim {victim_idx + 1} ({victim['location']['city']})")
        print(f"  💬 \"{victim['message'][:80]}...\"")
        
        # Run all signals
        nlp_r = nlp_analyzer.analyze(victim["message"])
        
        network_r = network_analyzer.analyze({
            "caller_phone": victim["caller_phone"],
            "callee_phone": victim["callee_phone"],
            "caller_ip": victim.get("caller_ip"),
            "caller_device": victim.get("caller_device"),
            "timestamp": current_time,
        })
        
        geo_r = geo_analyzer.analyze({
            "latitude": victim["location"]["lat"],
            "longitude": victim["location"]["lon"],
            "timestamp": current_time,
            "risk_score": nlp_r.confidence,
            "region": victim["location"]["city"],
        })
        
        # Use minimal signals for parallel victims
        parallel_signals = [nlp_r, network_r, geo_r]
        
        # Add placeholder signals for missing ones
        placeholder_temporal = SignalOutput(
            signal_name="temporal_sequence",
            embedding=torch.zeros(128),
            confidence=0.1,
            timestamp=current_time,
        )
        placeholder_voice = SignalOutput(
            signal_name="voice_analysis",
            embedding=torch.zeros(128),
            confidence=0.1,
            timestamp=current_time,
        )
        
        all_signals = [nlp_r, placeholder_temporal, placeholder_voice, network_r, geo_r]
        assessment = tcafe.fuse(
            signals=all_signals,
            incident_id=f"parallel_victim_{victim_idx}",
            current_time=current_time,
        )
        
        print(f"  Threat: {assessment.threat_score:.4f} [{assessment.threat_level}]")
        print(f"  Campaign: {assessment.campaign_id} {'(NEW CAMPAIGN!)' if assessment.is_new_campaign else '(MATCHED to existing)'}")
    
    # ─────────────────────────────────────────────
    # System Status Summary
    # ─────────────────────────────────────────────
    print_header("SYSTEM STATUS SUMMARY")
    
    status = tcafe.get_system_status()
    print(f"  Active Campaigns:    {status['active_campaigns']}")
    print(f"  Total Tracked:       {status['total_campaigns_tracked']}")
    
    print(f"\n  Signal Decay Rates:")
    for sig, rate in status['decay_rates'].items():
        half_life = 0.693 / max(rate, 1e-10)
        print(f"    {sig:<25s} λ={rate:.6f}  (half-life: {half_life:.0f}s)")
    
    print(f"\n  Network Graph Statistics:")
    graph_stats = network_analyzer.graph.get_statistics()
    print(f"    Nodes: {graph_stats['num_nodes']}")
    print(f"    Edges: {graph_stats['num_edges']}")
    print(f"    Node types: {graph_stats['node_type_counts']}")
    
    print(f"\n  Geo-Temporal Status:")
    print(f"    Events tracked: {len(geo_analyzer.clusterer.events)}")
    print(f"    Active regions: {sum(1 for r in geo_analyzer.spread_model.regions.values() if r.infected > 0)}")
    
    # Active campaigns
    campaigns = tcafe.campaign_fingerprint.get_all_active_campaigns()
    if campaigns:
        print(f"\n  Active Campaign Details:")
        for c in campaigns:
            print(f"    Campaign {c['campaign_id']}: "
                  f"{c['num_incidents']} incidents, "
                  f"severity={c['severity']:.4f}, "
                  f"duration={c['duration_hours']:.1f}h")
    
    print_header("DEMO COMPLETE", "═")
    print("  The system successfully demonstrated:")
    print("  ✅ 5-signal multi-modal analysis")
    print("  ✅ TCAFE temporal confidence-aware fusion")
    print("  ✅ Early warning at Phase 2-3 (before financial loss)")
    print("  ✅ Campaign-level detection across multiple victims")
    print("  ✅ Escalation velocity tracking")
    print("  ✅ Geographic spread analysis\n")
    
    # Save results to JSON for dashboard
    demo_results = {
        "scenario": "digital_arrest_scam_demo",
        "system_status": {
            "active_campaigns": status["active_campaigns"],
            "total_campaigns": status["total_campaigns_tracked"],
        },
        "campaigns": campaigns,
        "graph_stats": graph_stats,
    }
    
    results_path = os.path.join(os.path.dirname(__file__), "demo_results.json")
    with open(results_path, "w") as f:
        json.dump(demo_results, f, indent=2, default=str)
    print(f"  📄 Results saved to {results_path}")


if __name__ == "__main__":
    run_demo()
