"""
Voice Analysis Analyzer (Speaker Profiler) — Signal 3.

Combines all voice analysis sub-components into a single signal analyzer:
  - VoiceFeatureExtractor: Low-level acoustic features
  - DeepfakeDetector: Synthetic voice detection
  - EnvironmentClassifier: Call center / background detection

This signal answers: "Is this a real person? Are they reading a script?
Are they calling from a call center?"
"""

import time
from typing import Any, Dict, Optional

import torch
import numpy as np

from src.signals.base_signal import BaseSignalAnalyzer, SignalOutput
from src.signals.voice_analysis.feature_extractor import VoiceFeatureExtractor
from src.signals.voice_analysis.deepfake_detector import DeepfakeDetector
from src.signals.voice_analysis.environment_classifier import EnvironmentClassifier


class VoiceAnalysisAnalyzer(BaseSignalAnalyzer):
    """Signal 3: Voice Characteristics Analyzer.
    
    Analyzes audio for deepfake detection, stress patterns,
    and environmental classification.
    """
    
    def __init__(self, config: Dict[str, Any], device: Optional[str] = None):
        super().__init__(config, device)
        
        sample_rate = config.get("sample_rate", 16000)
        n_mfcc = config.get("n_mfcc", 40)
        n_fft = config.get("n_fft", 2048)
        hop_length = config.get("hop_length", 512)
        
        # Sub-components
        self.feature_extractor = VoiceFeatureExtractor(
            sample_rate=sample_rate, n_mfcc=n_mfcc,
            n_fft=n_fft, hop_length=hop_length,
        )
        
        self.deepfake_detector = DeepfakeDetector(
            n_mels=config.get("mel_bands", 128),
            prosody_features_dim=20,
        ).to(self.device)
        
        self.environment_classifier = EnvironmentClassifier(
            input_dim=128,
        ).to(self.device)
        
        # Embedding projection
        self.embedding_projection = torch.nn.Sequential(
            torch.nn.Linear(128 + 5, 128),  # deepfake emb + env probs
            torch.nn.ReLU(),
            torch.nn.Linear(128, config.get("embedding_dim", 128)),
        ).to(self.device)
        
        self.deepfake_detector.eval()
        self.environment_classifier.eval()
        self._initialized = True
    
    def get_signal_name(self) -> str:
        return "voice_analysis"
    
    def get_embedding_dim(self) -> int:
        return self.config.get("embedding_dim", 128)
    
    def get_decay_rate(self) -> float:
        return self.config.get("confidence_decay_rate", 0.03)
    
    def analyze(self, data: Any) -> SignalOutput:
        """Analyze audio for voice characteristics.
        
        Args:
            data: Dict with:
                - 'waveform': np.ndarray — mono audio (16kHz, normalized [-1,1])
                - 'sample_rate': int (optional, defaults to 16000)
              OR:
                - 'features': pre-extracted features dict (for testing)
                
        Returns:
            SignalOutput with deepfake probability, environment, and stress metrics
        """
        if isinstance(data, dict):
            waveform = data.get("waveform")
            if waveform is None and "features" in data:
                return self._analyze_from_features(data["features"])
            if waveform is None:
                waveform = np.zeros(16000, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            waveform = data
        else:
            waveform = np.zeros(16000, dtype=np.float32)
        
        # Step 1: Extract acoustic features
        features = self.feature_extractor.extract_all_features(waveform)
        
        # Step 2: Prepare mel spectrogram for CNN
        mel_spec = features["mel_spectrogram"]
        mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Step 3: Prepare prosody features for deepfake detection
        prosody_features = self._prepare_prosody_features(features)
        prosody_tensor = prosody_features.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Step 4: Deepfake detection
            df_logits, deepfake_prob, deepfake_emb = self.deepfake_detector(
                mel_tensor, prosody_tensor
            )
            
            # Step 5: Environment classification (from spectral features)
            noise_features = deepfake_emb  # Reuse deepfake embedding as proxy
            env_logits, env_probs = self.environment_classifier(noise_features)
            
            # Step 6: Create combined embedding
            combined = torch.cat([deepfake_emb.squeeze(0), env_probs.squeeze(0)], dim=-1)
            embedding = self.embedding_projection(combined.unsqueeze(0)).squeeze(0)
        
        # Build metadata
        env_idx = torch.argmax(env_probs, dim=-1).item()
        env_label = EnvironmentClassifier.ENV_LABELS[env_idx]
        
        metadata = {
            "deepfake_probability": round(deepfake_prob.item(), 4),
            "is_likely_deepfake": deepfake_prob.item() > self.config.get("deepfake_threshold", 0.7),
            "environment": env_label,
            "environment_confidence": round(env_probs[0, env_idx].item(), 4),
            "jitter": round(float(features.get("jitter", 0.0)), 6),
            "shimmer": round(float(features.get("shimmer", 0.0)), 6),
            "hnr": round(float(features.get("hnr", 0.0)), 4),
            "speech_rate": round(float(features.get("speech_rate", 0.0)), 4),
            "pause_features": features.get("pause_features", {}),
            "pitch_mean": round(float(np.mean(features["pitch_contour"][features["pitch_contour"] > 0]))
                                if np.any(features["pitch_contour"] > 0) else 0.0, 2),
        }
        
        # Confidence: higher if deepfake/environment detection is certain
        df_conf = abs(deepfake_prob.item() - 0.5) * 2  # Distance from uncertainty
        env_conf = env_probs[0, env_idx].item()
        confidence = 0.6 * df_conf + 0.4 * env_conf
        
        return self._create_output(
            embedding=embedding.cpu(),
            confidence=confidence,
            metadata=metadata,
        )
    
    def _prepare_prosody_features(self, features: Dict) -> torch.Tensor:
        """Prepare prosody features vector for deepfake detector."""
        prosody = []
        
        # Pitch features
        pitch = features.get("pitch_contour", np.array([0.0]))
        voiced = pitch[pitch > 0]
        if len(voiced) > 0:
            prosody.extend([
                float(np.mean(voiced)), float(np.std(voiced)),
                float(np.max(voiced) - np.min(voiced)),
                float(np.median(voiced)),
            ])
        else:
            prosody.extend([0.0, 0.0, 0.0, 0.0])
        
        # Voice quality
        prosody.append(float(features.get("jitter", 0.0)))
        prosody.append(float(features.get("shimmer", 0.0)))
        prosody.append(float(features.get("hnr", 0.0)))
        
        # Speech rate and pauses
        prosody.append(float(features.get("speech_rate", 0.0)))
        pause_feats = features.get("pause_features", {})
        prosody.extend([
            float(pause_feats.get("num_pauses", 0)),
            float(pause_feats.get("mean_pause_duration", 0.0)),
            float(pause_feats.get("pause_ratio", 0.0)),
        ])
        
        # Spectral statistics
        for key in ["spectral_centroid", "spectral_bandwidth", "zero_crossing_rate"]:
            arr = features.get(key, np.array([0.0]))
            prosody.extend([float(np.mean(arr)), float(np.std(arr))])
        
        # Energy statistics
        energy = features.get("energy_contour", np.array([0.0]))
        prosody.extend([float(np.mean(energy)), float(np.std(energy))])
        
        # Pad to expected dimension
        while len(prosody) < 20:
            prosody.append(0.0)
        
        return torch.tensor(prosody[:20], dtype=torch.float32)
    
    def _analyze_from_features(self, features: Dict) -> SignalOutput:
        """Create output from pre-extracted features (for testing)."""
        embedding = torch.randn(self.get_embedding_dim())
        return self._create_output(
            embedding=embedding,
            confidence=0.5,
            metadata=features,
        )
