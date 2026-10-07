"""
Scam Phase State Machine (SPSM) — A novel probabilistic state machine
that models the temporal progression of a digital arrest scam.

Instead of analyzing individual messages, the SPSM tracks the *sequence*
of scam phases, making the temporal ordering itself a signal.

The 6 phases of a digital arrest scam:
  Phase 1: AUTHORITY_CLAIM     — "This is CBI/Police calling"
  Phase 2: ACCUSATION          — "Your Aadhaar is linked to money laundering"
  Phase 3: FEAR_ESCALATION     — "Arrest warrant has been issued"
  Phase 4: ISOLATION           — "Don't tell anyone or you'll be arrested"
  Phase 5: FINANCIAL_EXTRACTION— "Transfer to this safe account"
  Phase 6: CONTINUED_CONTROL   — "Stay on the call, don't disconnect"

Key innovation: The transition probabilities between phases are LEARNED
from data, and deviations from typical scam sequences reduce confidence.
Early detection is possible because reaching Phase 2-3 with high transition
probability is sufficient to flag the threat — we don't need to wait
until Phase 5 (financial loss).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import IntEnum


class ScamPhase(IntEnum):
    """The 6 canonical phases of a digital arrest scam."""
    INITIAL = 0              # Pre-scam / normal conversation
    AUTHORITY_CLAIM = 1      # Establishing fake authority
    ACCUSATION = 2           # Making false accusations
    FEAR_ESCALATION = 3      # Escalating threats
    ISOLATION = 4            # Isolating the victim
    FINANCIAL_EXTRACTION = 5 # Extracting money
    CONTINUED_CONTROL = 6    # Maintaining control post-extraction


class ScamPhaseStateMachine(nn.Module):
    """Probabilistic state machine for scam phase tracking.
    
    Unlike a deterministic FSM, this model learns transition probabilities
    between phases and can handle:
      - Phase skipping (scammer jumps from Phase 1 → Phase 3)
      - Phase repetition (scammer returns to Phase 3 after Phase 4)
      - Partial sequences (early detection before all phases observed)
    
    Architecture:
      Current phase embedding + message features → transition probabilities
      
    The model outputs:
      - Current phase (argmax of phase probabilities)
      - Phase confidence (probability of the predicted phase)
      - Transition velocity (how fast phases are progressing)
      - Sequence anomaly score (how much does this deviate from typical scam sequences)
    """
    
    NUM_PHASES = 7  # 0 (initial) + 6 scam phases
    
    # Canonical transition matrix (prior knowledge, used for initialization)
    # Row = from phase, Col = to phase. High values = likely transitions.
    CANONICAL_TRANSITIONS = np.array([
        # INIT  AUTH  ACCUS FEAR  ISOL  FIN   CTRL
        [0.70,  0.25, 0.03, 0.01, 0.00, 0.00, 0.01],  # From INITIAL
        [0.05,  0.15, 0.65, 0.10, 0.03, 0.01, 0.01],  # From AUTHORITY
        [0.02,  0.05, 0.15, 0.60, 0.13, 0.04, 0.01],  # From ACCUSATION
        [0.01,  0.02, 0.05, 0.20, 0.50, 0.20, 0.02],  # From FEAR
        [0.01,  0.01, 0.02, 0.10, 0.15, 0.65, 0.06],  # From ISOLATION
        [0.01,  0.01, 0.01, 0.05, 0.05, 0.27, 0.60],  # From FINANCIAL
        [0.02,  0.01, 0.01, 0.05, 0.05, 0.26, 0.60],  # From CONTROL
    ], dtype=np.float32)
    
    def __init__(self, feature_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Phase embeddings
        self.phase_embeddings = nn.Embedding(self.NUM_PHASES, hidden_dim)
        
        # Transition network: predicts next phase from current state + features
        self.transition_network = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.NUM_PHASES),
        )
        
        # Phase classifier: directly classifies the phase from features
        self.phase_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.NUM_PHASES),
        )
        
        # Learned transition matrix (initialized from canonical)
        self.transition_prior = nn.Parameter(
            torch.from_numpy(self.CANONICAL_TRANSITIONS),
            requires_grad=True,
        )
        
        # Phase history tracking (not a parameter, runtime state)
        self._phase_history: List[int] = []
        self._phase_timestamps: List[float] = []
    
    def forward(
        self,
        features: torch.Tensor,
        current_phase: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict the current scam phase and transition probabilities.
        
        Args:
            features: (batch, feature_dim) — message/utterance features
            current_phase: (batch,) — current phase index (optional, for transition prediction)
            
        Returns:
            Dictionary with:
              - phase_logits: (batch, NUM_PHASES) — raw phase predictions
              - phase_probs: (batch, NUM_PHASES) — softmax probabilities
              - predicted_phase: (batch,) — argmax phase index
              - transition_probs: (batch, NUM_PHASES) — transition probabilities if current_phase given
        """
        batch_size = features.shape[0]
        
        # Direct phase classification
        phase_logits = self.phase_classifier(features)
        phase_probs = torch.softmax(phase_logits, dim=-1)
        predicted_phase = torch.argmax(phase_probs, dim=-1)
        
        result = {
            "phase_logits": phase_logits,
            "phase_probs": phase_probs,
            "predicted_phase": predicted_phase,
        }
        
        # If we know the current phase, predict transitions
        if current_phase is not None:
            phase_emb = self.phase_embeddings(current_phase)
            combined = torch.cat([phase_emb, features], dim=-1)
            transition_logits = self.transition_network(combined)
            
            # Blend learned transitions with prior knowledge
            prior = self.transition_prior[current_phase]
            blended = 0.7 * torch.softmax(transition_logits, dim=-1) + 0.3 * prior
            
            result["transition_probs"] = blended
        
        return result
    
    def update_history(self, phase: int, timestamp: float):
        """Track phase progression over time."""
        self._phase_history.append(phase)
        self._phase_timestamps.append(timestamp)
    
    def get_transition_velocity(self) -> float:
        """Compute how fast the scam is progressing through phases.
        
        Returns transitions per minute. Higher = faster escalation = more dangerous.
        """
        if len(self._phase_history) < 2:
            return 0.0
        
        transitions = sum(
            1 for i in range(1, len(self._phase_history))
            if self._phase_history[i] != self._phase_history[i - 1]
        )
        
        time_span = self._phase_timestamps[-1] - self._phase_timestamps[0]
        if time_span <= 0:
            return 0.0
        
        return transitions / (time_span / 60.0)  # transitions per minute
    
    def get_sequence_anomaly_score(self) -> float:
        """How much does the observed sequence deviate from canonical scam patterns?
        
        Low anomaly = follows typical scam script (more likely to be a real scam)
        High anomaly = unusual sequence (could be legitimate or novel scam variant)
        """
        if len(self._phase_history) < 2:
            return 0.0
        
        canonical = self.CANONICAL_TRANSITIONS
        log_likelihood = 0.0
        
        for i in range(1, len(self._phase_history)):
            from_phase = self._phase_history[i - 1]
            to_phase = self._phase_history[i]
            prob = canonical[from_phase][to_phase]
            log_likelihood += np.log(max(prob, 1e-10))
        
        # Normalize by sequence length
        avg_ll = log_likelihood / (len(self._phase_history) - 1)
        
        # Convert to anomaly score (0 = perfectly canonical, 1 = highly anomalous)
        anomaly = 1.0 - np.exp(avg_ll)
        return max(0.0, min(1.0, anomaly))
    
    def get_early_warning_score(self) -> float:
        """Compute early warning score based on phase progression.
        
        The key insight: if we've seen Phases 1→2→3 with canonical transitions,
        there's a high probability of reaching Phase 5 (financial loss).
        We can warn EARLY at Phase 2-3 instead of waiting until Phase 5.
        """
        if not self._phase_history:
            return 0.0
        
        current = self._phase_history[-1]
        velocity = self.get_transition_velocity()
        anomaly = self.get_sequence_anomaly_score()
        
        # Higher phase + faster velocity + lower anomaly = higher warning
        phase_factor = current / (self.NUM_PHASES - 1)  # 0 to 1
        velocity_factor = min(1.0, velocity / 2.0)       # Normalize velocity
        canonical_factor = 1.0 - anomaly                  # Following the script
        
        warning = 0.4 * phase_factor + 0.35 * velocity_factor + 0.25 * canonical_factor
        return max(0.0, min(1.0, warning))
    
    def reset(self):
        """Reset state machine for a new conversation."""
        self._phase_history = []
        self._phase_timestamps = []
