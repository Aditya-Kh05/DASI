"""
Environment Classifier — Identify the acoustic environment of a call.

Scam call centers have distinctive acoustic signatures:
  - Multiple speakers in background (other scammers)
  - Echo patterns from small rooms
  - Consistent HVAC/fan noise
  - Keyboard typing sounds
  - Similar acoustic fingerprints across multiple calls

This module classifies the background environment to add context
to the voice analysis signal.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class EnvironmentClassifier(nn.Module):
    """Classify the acoustic environment from background noise features.
    
    Environment categories:
      0: CLEAN         — quiet environment (home/office)
      1: CALL_CENTER   — multiple background voices, typing
      2: OUTDOOR       — traffic, wind, ambient noise
      3: VEHICLE       — engine noise, road sounds
      4: NOISY_ROOM    — TV, music, crowd noise
    """
    
    ENV_LABELS = ["CLEAN", "CALL_CENTER", "OUTDOOR", "VEHICLE", "NOISY_ROOM"]
    NUM_CLASSES = 5
    
    def __init__(self, input_dim: int = 128):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.NUM_CLASSES),
        )
    
    def forward(self, noise_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classify environment type.
        
        Args:
            noise_features: (batch, input_dim) — extracted noise features
            
        Returns:
            Tuple of:
              - env_logits: (batch, NUM_CLASSES)
              - env_probs: (batch, NUM_CLASSES) — softmax probabilities
        """
        logits = self.classifier(noise_features)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs
    
    def get_environment(self, noise_features: torch.Tensor) -> Dict:
        """Get the predicted environment with confidence."""
        with torch.no_grad():
            _, probs = self.forward(noise_features)
        
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_idx].item() if probs.dim() > 1 else probs[pred_idx].item()
        
        return {
            "environment": self.ENV_LABELS[pred_idx],
            "confidence": round(confidence, 4),
            "all_probabilities": {
                label: round(probs[0, i].item() if probs.dim() > 1 else probs[i].item(), 4)
                for i, label in enumerate(self.ENV_LABELS)
            },
        }
