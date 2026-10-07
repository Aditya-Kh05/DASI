"""
Escalation Velocity Tensor — TCAFE Component 3.

Measures the RATE OF CHANGE of the fused threat state across all dimensions.

Core insight: A rapidly increasing threat vector (even if absolute scores 
are currently low) is a strong early warning signal. This is the mechanism
that enables EARLY detection — we don't wait for the threat to be high,
we detect that it's RISING FAST.

Mathematical formulation:
  Φ'(t) = ∂Φ/∂t ≈ (Φ(t) - Φ(t-Δ)) / Δ
  V(t) = ‖Φ'(t)‖₂    (L2 norm — scalar velocity)
  
The full velocity TENSOR preserves directional information:
  Which signals are escalating? How fast? In which direction?

This is more informative than a scalar velocity because:
  - NLP escalating + Voice stable → specific threat pattern
  - All signals escalating simultaneously → campaign-level attack
  - Network escalating + Geo escalating → organized spread
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from collections import deque


class EscalationVelocityTensor(nn.Module):
    """Compute and analyze the rate of change of the fused threat state.
    
    Maintains a sliding window of historical threat states to compute:
      1. Instantaneous velocity: Φ'(t)
      2. Acceleration: Φ''(t)  
      3. Per-signal velocity decomposition
      4. Escalation alerts when velocity exceeds threshold
    """
    
    def __init__(
        self,
        state_dim: int = 256,
        window_size: int = 5,
        velocity_threshold: float = 0.3,
        acceleration_threshold: float = 0.5,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.window_size = window_size
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        
        # Historical state buffer
        self._state_history: deque = deque(maxlen=window_size + 1)
        self._time_history: deque = deque(maxlen=window_size + 1)
        
        # Learnable velocity weighting (which dimensions matter more for alerts)
        self.velocity_weights = nn.Parameter(
            torch.ones(state_dim) / state_dim
        )
        
        # Velocity embedding network (maps velocity tensor to a compact representation)
        self.velocity_encoder = nn.Sequential(
            nn.Linear(state_dim * 3, state_dim),  # velocity + acceleration + state
            nn.ReLU(),
            nn.Linear(state_dim, state_dim // 2),
            nn.ReLU(),
            nn.Linear(state_dim // 2, 64),
        )
        
        # Alert classifier
        self.alert_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # 4 alert levels: none, low, medium, high
        )
    
    def update(self, state: torch.Tensor, timestamp: float):
        """Add a new state to the history buffer.
        
        Args:
            state: (state_dim,) — current fused threat state Φ(t)
            timestamp: float — time of this state
        """
        self._state_history.append(state.detach().clone())
        self._time_history.append(timestamp)
    
    def compute_velocity(self) -> Dict[str, torch.Tensor]:
        """Compute the escalation velocity tensor.
        
        Returns:
            Dictionary with:
              - velocity: (state_dim,) — instantaneous velocity Φ'(t)
              - speed: scalar — L2 norm of velocity
              - acceleration: (state_dim,) — second derivative Φ''(t)
              - acceleration_magnitude: scalar
              - weighted_speed: scalar — velocity weighted by learned importance
              - direction: (state_dim,) — unit velocity vector (which dimensions are escalating)
        """
        if len(self._state_history) < 2:
            zero = torch.zeros(self.state_dim)
            return {
                "velocity": zero,
                "speed": torch.tensor(0.0),
                "acceleration": zero,
                "acceleration_magnitude": torch.tensor(0.0),
                "weighted_speed": torch.tensor(0.0),
                "direction": zero,
            }
        
        states = list(self._state_history)
        times = list(self._time_history)
        
        # Instantaneous velocity: Φ'(t) = (Φ(t) - Φ(t-1)) / Δt
        dt = max(times[-1] - times[-2], 1e-6)
        velocity = (states[-1] - states[-2]) / dt
        
        # Speed: ‖Φ'(t)‖₂
        speed = torch.norm(velocity)
        
        # Weighted speed (learned importance per dimension)
        weighted_speed = torch.sum(torch.abs(velocity) * torch.softmax(self.velocity_weights, dim=0))
        
        # Direction: unit vector showing which dimensions are escalating
        direction = velocity / (speed + 1e-10)
        
        # Acceleration: Φ''(t) = (Φ'(t) - Φ'(t-1)) / Δt
        if len(states) >= 3:
            dt_prev = max(times[-2] - times[-3], 1e-6)
            velocity_prev = (states[-2] - states[-3]) / dt_prev
            acceleration = (velocity - velocity_prev) / dt
            acceleration_magnitude = torch.norm(acceleration)
        else:
            acceleration = torch.zeros(self.state_dim)
            acceleration_magnitude = torch.tensor(0.0)
        
        return {
            "velocity": velocity,
            "speed": speed,
            "acceleration": acceleration,
            "acceleration_magnitude": acceleration_magnitude,
            "weighted_speed": weighted_speed,
            "direction": direction,
        }
    
    def get_velocity_embedding(self, state: torch.Tensor) -> torch.Tensor:
        """Encode the velocity information into a compact embedding.
        
        Args:
            state: (state_dim,) — current state
            
        Returns:
            (64,) — velocity embedding for the campaign fingerprint
        """
        vel_info = self.compute_velocity()
        
        combined = torch.cat([
            state,
            vel_info["velocity"],
            vel_info["acceleration"],
        ])
        
        with torch.no_grad():
            embedding = self.velocity_encoder(combined.unsqueeze(0)).squeeze(0)
        
        return embedding
    
    def check_alert(self) -> Dict:
        """Check if escalation velocity triggers an alert.
        
        Returns:
            Alert information with level and supporting data
        """
        vel_info = self.compute_velocity()
        speed = vel_info["speed"].item()
        accel_mag = vel_info["acceleration_magnitude"].item()
        weighted_speed = vel_info["weighted_speed"].item()
        
        # Determine alert level
        if speed > self.velocity_threshold * 3 or accel_mag > self.acceleration_threshold * 2:
            alert_level = "CRITICAL"
            alert_score = 1.0
        elif speed > self.velocity_threshold * 2 or accel_mag > self.acceleration_threshold:
            alert_level = "HIGH"
            alert_score = 0.8
        elif speed > self.velocity_threshold:
            alert_level = "MEDIUM"
            alert_score = 0.5
        elif speed > self.velocity_threshold * 0.5:
            alert_level = "LOW"
            alert_score = 0.3
        else:
            alert_level = "NONE"
            alert_score = 0.0
        
        # Find top escalating dimensions
        velocity = vel_info["velocity"]
        top_dims = torch.topk(torch.abs(velocity), min(5, len(velocity)))
        
        return {
            "alert_level": alert_level,
            "alert_score": round(alert_score, 4),
            "speed": round(speed, 6),
            "acceleration": round(accel_mag, 6),
            "weighted_speed": round(weighted_speed, 6),
            "top_escalating_dimensions": top_dims.indices.tolist(),
            "top_escalation_values": [round(v, 6) for v in top_dims.values.tolist()],
            "history_length": len(self._state_history),
        }
    
    def get_trend(self, lookback: int = 5) -> Dict:
        """Analyze the recent trend of the threat state.
        
        Returns whether the threat is increasing, decreasing, or stable.
        """
        if len(self._state_history) < 2:
            return {"trend": "UNKNOWN", "slope": 0.0}
        
        states = list(self._state_history)[-lookback:]
        norms = [torch.norm(s).item() for s in states]
        
        if len(norms) < 2:
            return {"trend": "STABLE", "slope": 0.0}
        
        # Simple linear regression on norms
        x = list(range(len(norms)))
        x_mean = sum(x) / len(x)
        y_mean = sum(norms) / len(norms)
        
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, norms))
        denominator = sum((xi - x_mean) ** 2 for xi in x)
        
        slope = numerator / max(denominator, 1e-10)
        
        if slope > 0.01:
            trend = "ESCALATING"
        elif slope < -0.01:
            trend = "DE_ESCALATING"
        else:
            trend = "STABLE"
        
        return {
            "trend": trend,
            "slope": round(slope, 6),
            "recent_norms": [round(n, 4) for n in norms],
        }
    
    def reset(self):
        """Clear history for a new session."""
        self._state_history.clear()
        self._time_history.clear()
