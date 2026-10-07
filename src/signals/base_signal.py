"""
Base Signal Analyzer — Abstract interface for all 5 signal analyzers.

Every signal in the system produces a standardized SignalOutput:
  - embedding: dense vector representation
  - confidence: reliability score [0, 1]
  - timestamp: when the signal was computed
  - metadata: signal-specific additional information

This abstraction enables the TCAFE fusion engine to treat all signals
uniformly while allowing each signal to implement its own analysis logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import math

import torch
import numpy as np


@dataclass
class SignalOutput:
    """Standardized output from any signal analyzer.
    
    This is the contract between signal analyzers and the TCAFE fusion engine.
    Every signal must produce this format regardless of its internal complexity.
    """
    
    signal_name: str
    embedding: torch.Tensor
    confidence: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age(self) -> float:
        """How old this signal is in seconds."""
        return time.time() - self.timestamp
    
    def decayed_confidence(self, decay_rate: float, current_time: Optional[float] = None) -> float:
        """Compute temporally-decayed confidence.
        
        Core TCAFE mechanism — signals become less reliable over time.
        Formula: c̃ = c · exp(-λ · age)
        """
        t_now = current_time or time.time()
        age = max(0.0, t_now - self.timestamp)
        return self.confidence * math.exp(-decay_rate * age)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API responses and logging."""
        def _clean(val):
            if isinstance(val, dict):
                return {k: _clean(v) for k, v in val.items()}
            elif isinstance(val, (list, tuple)):
                return [_clean(v) for v in val]
            elif isinstance(val, (np.floating, float)):
                return float(val)
            elif isinstance(val, (np.integer, int)):
                return int(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            elif isinstance(val, torch.Tensor):
                return val.detach().cpu().tolist()
            return val

        return {
            "signal_name": self.signal_name,
            "embedding_shape": list(self.embedding.shape),
            "embedding_norm": float(torch.norm(self.embedding).item()),
            "confidence": round(float(self.confidence), 4),
            "timestamp": float(self.timestamp),
            "age_seconds": round(float(self.age), 2),
            "metadata": _clean(self.metadata),
        }


class BaseSignalAnalyzer(ABC):
    """Abstract base class for all signal analyzers.
    
    Each signal analyzer extends this class and implements:
      - analyze(): Core analysis logic producing a SignalOutput
      - get_signal_name(): Unique identifier for this signal type
      - get_embedding_dim(): Dimension of the output embedding
      - get_decay_rate(): How quickly this signal's confidence degrades
    """
    
    def __init__(self, config: Dict[str, Any], device: Optional[str] = None):
        self.config = config
        self.device = self._resolve_device(device)
        self._initialized = False
    
    @staticmethod
    def _resolve_device(device: Optional[str]) -> torch.device:
        if device is None or device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    @abstractmethod
    def get_signal_name(self) -> str:
        ...
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        ...
    
    @abstractmethod
    def get_decay_rate(self) -> float:
        ...
    
    @abstractmethod
    def analyze(self, data: Any) -> SignalOutput:
        ...
    
    def _estimate_confidence(self, logits: torch.Tensor) -> float:
        """Estimate confidence from model logits via entropy."""
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        max_entropy = math.log(logits.shape[-1])
        confidence = 1.0 - (entropy.item() / max_entropy)
        return max(0.0, min(1.0, confidence))
    
    def _create_output(
        self, embedding: torch.Tensor, confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SignalOutput:
        return SignalOutput(
            signal_name=self.get_signal_name(),
            embedding=embedding.detach().cpu(),
            confidence=max(0.0, min(1.0, confidence)),
            timestamp=time.time(),
            metadata=metadata or {},
        )
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"signal={self.get_signal_name()}, "
            f"dim={self.get_embedding_dim()}, "
            f"decay={self.get_decay_rate()}, "
            f"device={self.device})"
        )
