"""
Temporal Sequence Analyzer — Signal 2 (Phase Detector).

This signal answers: "What phase of the scam are we in, and how fast is it progressing?"

Instead of analyzing individual sentences, this module understands the SEQUENCE:
  Phase 1 → Phase 2 → Phase 3 → ...

The order itself becomes a signal. A message saying "transfer money" is much more
suspicious if preceded by "I am from CBI" and "your Aadhaar is compromised"
than if it appears in isolation.

Key innovation: Early detection at Phase 2-3 before financial loss at Phase 5.
"""

import time
from typing import Any, Dict, List, Optional

import torch
import numpy as np

from src.signals.base_signal import BaseSignalAnalyzer, SignalOutput
from src.signals.temporal_sequence.state_machine import ScamPhaseStateMachine, ScamPhase
from src.signals.temporal_sequence.sequence_encoder import SequenceEncoder


class TemporalSequenceAnalyzer(BaseSignalAnalyzer):
    """Signal 2: Temporal Scam Sequence Detector.
    
    Tracks the progression of a conversation through scam phases
    and computes early warning scores based on transition patterns.
    """
    
    PHASE_NAMES = {
        0: "INITIAL",
        1: "AUTHORITY_CLAIM",
        2: "ACCUSATION",
        3: "FEAR_ESCALATION",
        4: "ISOLATION",
        5: "FINANCIAL_EXTRACTION",
        6: "CONTINUED_CONTROL",
    }
    
    def __init__(self, config: Dict[str, Any], device: Optional[str] = None):
        super().__init__(config, device)
        
        embedding_dim = config.get("embedding_dim", 128)
        
        # State machine for phase tracking
        self.state_machine = ScamPhaseStateMachine(
            feature_dim=256,
            hidden_dim=embedding_dim,
        ).to(self.device)
        
        # Sequence encoder for temporal embedding
        self.sequence_encoder = SequenceEncoder(
            input_dim=256,
            hidden_dim=256,
            num_layers=config.get("lstm_num_layers", 2),
            output_dim=embedding_dim,
            dropout=config.get("dropout", 0.2),
        ).to(self.device)
        
        # Feature projection (to match state machine input)
        self.feature_projection = torch.nn.Linear(256, 256).to(self.device)
        
        # Message history for sequence encoding
        self._message_embeddings: List[torch.Tensor] = []
        
        self.state_machine.eval()
        self.sequence_encoder.eval()
        self._initialized = True
    
    def get_signal_name(self) -> str:
        return "temporal_sequence"
    
    def get_embedding_dim(self) -> int:
        return self.config.get("embedding_dim", 128)
    
    def get_decay_rate(self) -> float:
        return self.config.get("confidence_decay_rate", 0.05)
    
    def analyze(self, data: Any) -> SignalOutput:
        """Analyze the temporal progression of a conversation.
        
        Args:
            data: Dict with:
                - 'message_embedding': torch.Tensor (256,) from conversation semantics
                - 'message_index': int — position in conversation
                - 'timestamp': float — optional message timestamp
                
        Returns:
            SignalOutput with phase detection and early warning metadata
        """
        if isinstance(data, dict):
            msg_embedding = data.get("message_embedding")
            if msg_embedding is None:
                msg_embedding = data.get("embedding")
            if msg_embedding is None:
                msg_embedding = torch.zeros(self.config.get("input_dim", 256))
            timestamp = data.get("timestamp", time.time())
        else:
            msg_embedding = torch.zeros(self.config.get("input_dim", 256))
            timestamp = time.time()
        
        if isinstance(msg_embedding, np.ndarray):
            msg_embedding = torch.from_numpy(msg_embedding).float()
        
        msg_embedding = msg_embedding.to(self.device)
        if msg_embedding.dim() == 1:
            msg_embedding = msg_embedding.unsqueeze(0)
        
        # Store in history
        self._message_embeddings.append(msg_embedding.detach().cpu())
        
        with torch.no_grad():
            # Step 1: Detect current phase
            features = self.feature_projection(msg_embedding)
            
            current_phase_idx = None
            if self.state_machine._phase_history:
                current_phase_idx = torch.tensor(
                    [self.state_machine._phase_history[-1]], 
                    device=self.device
                )
            
            phase_result = self.state_machine(features, current_phase_idx)
            predicted_phase = phase_result["predicted_phase"].item()
            phase_confidence = phase_result["phase_probs"][0, predicted_phase].item()
            
            # Step 2: Update state machine history
            self.state_machine.update_history(predicted_phase, timestamp)
            
            # Step 3: Encode full sequence so far
            if len(self._message_embeddings) >= 2:
                seq_tensor = torch.cat(self._message_embeddings, dim=0).unsqueeze(0).to(self.device)
                temporal_embedding, attention_weights = self.sequence_encoder(seq_tensor)
                temporal_embedding = temporal_embedding.squeeze(0)
                attn = attention_weights.squeeze(0).cpu().numpy().tolist()
            else:
                temporal_embedding = torch.zeros(self.get_embedding_dim(), device=self.device)
                attn = [1.0]
            
            # Step 4: Compute scam progression metrics
            transition_velocity = self.state_machine.get_transition_velocity()
            sequence_anomaly = self.state_machine.get_sequence_anomaly_score()
            early_warning = self.state_machine.get_early_warning_score()
        
        # Build metadata
        metadata = {
            "current_phase": predicted_phase,
            "phase_name": self.PHASE_NAMES.get(predicted_phase, "UNKNOWN"),
            "phase_confidence": round(phase_confidence, 4),
            "phase_history": list(self.state_machine._phase_history),
            "transition_velocity": round(transition_velocity, 4),
            "sequence_anomaly_score": round(sequence_anomaly, 4),
            "early_warning_score": round(early_warning, 4),
            "messages_analyzed": len(self._message_embeddings),
            "attention_over_messages": [round(a, 4) for a in attn[-10:]],  # Last 10
        }
        
        # Confidence combines phase detection certainty with sequence evidence
        confidence = 0.5 * phase_confidence + 0.3 * early_warning + 0.2 * (1 - sequence_anomaly)
        
        return self._create_output(
            embedding=temporal_embedding.cpu(),
            confidence=confidence,
            metadata=metadata,
        )
    
    def reset(self):
        """Reset for a new conversation."""
        self.state_machine.reset()
        self._message_embeddings = []
