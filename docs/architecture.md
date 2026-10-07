# Architecture — Digital Arrest Scam Intelligence System

## System Overview

The system consists of three major layers:

1. **Signal Layer** — 5 independent signal analyzers that extract heterogeneous features
2. **Fusion Layer** — TCAFE engine that combines signals with temporal awareness  
3. **Intelligence Layer** — Campaign tracking, prediction, and alerting

## Data Flow

```
Input Event (call/message/transaction)
    │
    ├──→ Signal 1: Conversation Semantics (NLP)
    │         └── BERT encoder → 5 scam sub-scores → 256-dim embedding
    │
    ├──→ Signal 2: Temporal Sequence (Phase Detection)
    │         └── SPSM + LSTM → Phase ID + transition velocity → 128-dim embedding
    │
    ├──→ Signal 3: Voice Analysis (Audio)
    │         └── CNN (mel-spectrogram) + MLP (prosody) → deepfake + stress → 128-dim embedding
    │
    ├──→ Signal 4: Network Graph (Relationships)
    │         └── GraphSAGE on communication graph → node risk + cluster → 128-dim embedding
    │
    └──→ Signal 5: Geo-Temporal (Geography + Time)
              └── ST-DBSCAN + SIR model → hotspot + spread → 64-dim embedding
    
    All signals produce: (embedding, confidence, timestamp)
    │
    ▼
TCAFE Fusion Engine
    │
    ├── Step 1: Confidence Decay Gate     — c̃ᵢ = cᵢ · exp(-λᵢ · age)
    ├── Step 2: Cross-Signal Attention    — Multi-head attention across all signals
    ├── Step 3: State Fusion              — Weighted combination → Φ(t) ∈ ℝ²⁵⁶
    ├── Step 4: Escalation Velocity       — V(t) = ‖∂Φ/∂t‖₂
    ├── Step 5: Campaign Fingerprinting   — Match to known campaigns
    └── Step 6: Threat Assessment         — Final score + level + alert
    │
    ▼
Output: ThreatAssessment
    ├── threat_score: [0, 1]
    ├── threat_level: MINIMAL | LOW | MEDIUM | HIGH | CRITICAL
    ├── escalation_alert: NONE | LOW | MEDIUM | HIGH | CRITICAL
    ├── campaign_id: assigned campaign
    ├── signal_contributions: per-signal importance
    └── cross_signal_correlations: interaction matrix
```

## Key Technical Decisions

### Why TCAFE over Simple Fusion
- **Temporal decay**: Signals age at different rates
- **Cross-attention**: Captures signal interactions (reinforcement/dampening)
- **Velocity tensor**: Enables early detection via rate-of-change
- **Campaign fingerprinting**: Aggregates incidents into organized campaigns

### Why 5 Signals
Each signal captures a fundamentally different modality:
1. **What** is being said (semantics)
2. **When** it's being said (temporal order)
3. **How** it's being said (voice)
4. **Who** is saying it (network)
5. **Where** it's happening (geography)

No single signal is sufficient. The power is in their combination.

### Signal Independence
Each signal analyzer is independently deployable and testable.
If one signal is unavailable (e.g., no audio), the system gracefully
degrades by assigning low confidence to that signal.
