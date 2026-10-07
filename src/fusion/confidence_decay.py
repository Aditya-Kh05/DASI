"""
Confidence Decay Gate — TCAFE Component 1.

Implements temporal decay of signal confidence based on signal staleness.

Core insight: A signal computed 5 minutes ago is less reliable than one
computed just now. Different signal types decay at different rates:
  - NLP text analysis: slow decay (text doesn't change)
  - Network graph: slow decay (structure is stable)
  - Voice analysis: medium decay
  - Temporal sequence: medium decay
  - Geo-temporal: fast decay (patterns shift quickly)

Mathematical formulation:
  c̃ᵢ = cᵢ · exp(-λᵢ · (t_now - tᵢ))
  
where:
  cᵢ = raw confidence of signal i
  λᵢ = decay rate specific to signal type i
  tᵢ = timestamp when signal i was computed
  c̃ᵢ = effective (decayed) confidence
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.signals.base_signal import SignalOutput


class ConfidenceDecayGate(nn.Module):
    """Apply temporal confidence decay to signal outputs.
    
    This is a learnable gate that modulates signal contributions
    based on their temporal reliability.
    
    The decay rates can be:
      1. Fixed (from config) — used for known signal characteristics
      2. Learned (nn.Parameter) — fine-tuned during training
      3. Adaptive — adjusted based on signal quality history
    """
    
    # Default decay rates per signal type
    DEFAULT_DECAY_RATES = {
        "conversation_semantics": 0.01,   # Very slow — text is stable
        "temporal_sequence": 0.05,        # Medium — sequence state can change
        "voice_analysis": 0.03,           # Medium — audio features are per-call
        "network_graph": 0.005,           # Very slow — network structure is stable
        "geo_temporal": 0.10,             # Fast — geographic patterns shift
    }
    
    def __init__(self, signal_names: List[str], learnable: bool = True):
        super().__init__()
        
        self.signal_names = signal_names
        self.num_signals = len(signal_names)
        
        # Initialize decay rates
        initial_rates = [
            self.DEFAULT_DECAY_RATES.get(name, 0.05)
            for name in signal_names
        ]
        
        if learnable:
            # Learnable decay rates (in log space for positivity)
            self.log_decay_rates = nn.Parameter(
                torch.log(torch.tensor(initial_rates, dtype=torch.float32))
            )
        else:
            self.register_buffer(
                'log_decay_rates',
                torch.log(torch.tensor(initial_rates, dtype=torch.float32))
            )
        
        # Learnable minimum confidence floor (signals never go below this)
        self.confidence_floor = nn.Parameter(
            torch.tensor(0.05, dtype=torch.float32)
        )
    
    @property
    def decay_rates(self) -> torch.Tensor:
        """Get positive decay rates from log-space parameters."""
        return torch.exp(self.log_decay_rates)
    
    def forward(
        self,
        confidences: torch.Tensor,
        timestamps: torch.Tensor,
        current_time: Optional[float] = None,
    ) -> torch.Tensor:
        """Apply temporal decay to signal confidences.
        
        Args:
            confidences: (batch, num_signals) — raw confidence scores
            timestamps: (batch, num_signals) — signal computation timestamps
            current_time: Reference time (defaults to now)
            
        Returns:
            decayed_confidences: (batch, num_signals) — temporally-adjusted confidences
        """
        t_now = current_time or time.time()
        
        # Compute signal ages
        ages = t_now - timestamps  # (batch, num_signals)
        ages = torch.clamp(ages, min=0.0)
        
        # Apply exponential decay: c̃ = c · exp(-λ · age)
        decay_factors = torch.exp(-self.decay_rates.unsqueeze(0) * ages)
        
        # Apply decay with confidence floor
        decayed = confidences * decay_factors
        decayed = torch.clamp(decayed, min=self.confidence_floor.item())
        
        return decayed
    
    def get_decay_report(self, signals: List[SignalOutput], current_time: Optional[float] = None) -> Dict:
        """Generate a human-readable report of signal confidence decay.
        
        Useful for debugging and visualization.
        """
        t_now = current_time or time.time()
        report = {}
        
        for i, signal in enumerate(signals):
            signal_name = signal.signal_name
            decay_rate = self.decay_rates[i].item() if i < len(self.decay_rates) else 0.05
            age = t_now - signal.timestamp
            
            raw_conf = signal.confidence
            decayed_conf = raw_conf * math.exp(-decay_rate * max(0, age))
            decayed_conf = max(decayed_conf, self.confidence_floor.item())
            
            report[signal_name] = {
                "raw_confidence": round(raw_conf, 4),
                "decayed_confidence": round(decayed_conf, 4),
                "decay_rate": round(decay_rate, 6),
                "age_seconds": round(age, 2),
                "decay_factor": round(math.exp(-decay_rate * max(0, age)), 4),
                "half_life_seconds": round(math.log(2) / max(decay_rate, 1e-10), 1),
            }
        
        return report
