"""
Deepfake Voice Detector — Identifies AI-generated / synthetic voices.

Digital arrest scammers increasingly use AI voice cloning to impersonate
officials. This module detects synthetic speech by analyzing:

1. Spectral artifacts — AI voices have subtle spectral regularities
2. Biological markers — Missing micro-breathing, swallowing, lip smacks
3. Prosody abnormalities — Unnatural pitch transitions and rhythm
4. Phase coherence — GAN/TTS-generated audio has phase artifacts

Architecture:
  Mel spectrogram → CNN feature extractor → Binary classifier (real/fake)
  + Prosody features → MLP → Score
  → Weighted combination → Deepfake probability
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class SpectrogramCNN(nn.Module):
    """CNN for spectral artifact detection in mel spectrograms."""
    
    def __init__(self, n_mels: int = 128, cnn_channels: list = None):
        super().__init__()
        
        channels = cnn_channels or [1, 32, 64, 128, 256]
        
        layers = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ])
        
        self.conv_layers = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels[-1] * 4 * 4, 256)
    
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Extract spectral features from mel spectrogram.
        
        Args:
            mel_spec: (batch, 1, n_mels, n_frames) — mel spectrogram
            
        Returns:
            (batch, 256) — spectral feature vector
        """
        x = self.conv_layers(mel_spec)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class DeepfakeDetector(nn.Module):
    """Detect AI-generated voices using spectral and prosodic analysis.
    
    Combines:
      1. CNN on mel spectrograms (spectral artifacts)
      2. MLP on prosody features (jitter, shimmer, HNR, pitch stability)
      
    Output: probability that the voice is synthetic/deepfake
    """
    
    def __init__(self, n_mels: int = 128, prosody_features_dim: int = 20):
        super().__init__()
        
        # Spectral analysis path
        self.spectrogram_cnn = SpectrogramCNN(n_mels=n_mels)
        
        # Prosody analysis path
        self.prosody_mlp = nn.Sequential(
            nn.Linear(prosody_features_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        
        # Fusion and classification
        self.classifier = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [real, fake]
        )
        
        # Embedding output (for TCAFE)
        self.embedding_head = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
    
    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        prosody_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Detect deepfake voice.
        
        Args:
            mel_spectrogram: (batch, 1, n_mels, n_frames)
            prosody_features: (batch, prosody_dim) — jitter, shimmer, HNR, etc.
            
        Returns:
            Tuple of:
              - deepfake_logits: (batch, 2) — [real, fake] logits
              - deepfake_prob: (batch,) — probability of being deepfake
              - embedding: (batch, 128) — for TCAFE fusion
        """
        # Spectral path
        spectral_features = self.spectrogram_cnn(mel_spectrogram)
        
        # Prosody path
        prosody_enc = self.prosody_mlp(prosody_features)
        
        # Fuse
        combined = torch.cat([spectral_features, prosody_enc], dim=-1)
        
        # Classify
        logits = self.classifier(combined)
        probs = torch.softmax(logits, dim=-1)
        deepfake_prob = probs[:, 1]  # Probability of being fake
        
        # Embedding
        embedding = self.embedding_head(combined)
        
        return logits, deepfake_prob, embedding
