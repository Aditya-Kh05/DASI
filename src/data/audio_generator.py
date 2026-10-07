"""
Audio Feature Generator — Synthetic Audio Feature Generation.

Since we cannot distribute real audio recordings, this module generates
realistic synthetic audio feature tensors (mel-spectrograms, MFCCs,
prosody features) that match the statistical distributions of:
  - Real human speech
  - AI-generated deepfake speech
  - Various call environments (office, call center, street, home)

The generated features are directly consumable by the VoiceAnalysisAnalyzer.
"""

import random
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class GeneratedAudioFeatures:
    """Synthetic audio features for one call segment."""
    sample_id: str
    mel_spectrogram: torch.Tensor       # (1, n_mels, n_frames)
    mfcc: torch.Tensor                  # (n_mfcc, n_frames)
    prosody_features: Dict[str, float]  # jitter, shimmer, HNR, pitch, etc.
    is_deepfake: bool
    environment: str                     # "QUIET_ROOM", "CALL_CENTER", "STREET", "VOIP"
    labels: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "sample_id": self.sample_id,
            "mel_shape": list(self.mel_spectrogram.shape),
            "mfcc_shape": list(self.mfcc.shape),
            "prosody_features": self.prosody_features,
            "is_deepfake": self.is_deepfake,
            "environment": self.environment,
        }


class AudioFeatureGenerator:
    """Generates synthetic audio features with realistic distributions.
    
    Deepfake voice characteristics (based on ASVspoof literature):
      - Smoother mel-spectrograms (less natural variation)
      - Lower jitter and shimmer (unnaturally stable)
      - Artifacts in high-frequency bands
      - More uniform pitch contour
    
    Real voice characteristics:
      - Natural variation in mel-spectrograms
      - Higher jitter (0.003-0.01) and shimmer (0.04-0.12)
      - Natural harmonics-to-noise ratio (10-25 dB)
      - Natural pitch variation
    """
    
    # Statistical profiles for prosody features
    PROSODY_PROFILES = {
        "real_male": {
            "pitch_mean": (100, 150),    # Hz range
            "pitch_std": (15, 40),
            "jitter": (0.003, 0.010),
            "shimmer": (0.04, 0.12),
            "hnr": (12, 25),             # dB
            "speech_rate": (3.5, 5.5),   # syllables/sec
            "pause_ratio": (0.15, 0.30),
            "f0_slope": (-0.02, 0.02),
        },
        "real_female": {
            "pitch_mean": (170, 255),
            "pitch_std": (20, 50),
            "jitter": (0.003, 0.008),
            "shimmer": (0.03, 0.10),
            "hnr": (15, 28),
            "speech_rate": (3.8, 5.8),
            "pause_ratio": (0.12, 0.28),
            "f0_slope": (-0.02, 0.02),
        },
        "deepfake": {
            "pitch_mean": (110, 200),
            "pitch_std": (5, 15),        # Much lower variation (unnatural)
            "jitter": (0.001, 0.004),    # Too stable
            "shimmer": (0.01, 0.04),     # Too stable
            "hnr": (20, 35),             # Artificially clean
            "speech_rate": (4.0, 5.0),   # Very consistent
            "pause_ratio": (0.08, 0.15), # Fewer natural pauses
            "f0_slope": (-0.005, 0.005), # Very flat pitch contour
        },
    }
    
    # Environment noise profiles (SNR in dB)
    ENVIRONMENT_PROFILES = {
        "QUIET_ROOM": {"snr": (25, 40), "noise_type": "gaussian", "reverb": 0.1},
        "CALL_CENTER": {"snr": (10, 20), "noise_type": "babble", "reverb": 0.3},
        "STREET": {"snr": (5, 15), "noise_type": "traffic", "reverb": 0.05},
        "VOIP": {"snr": (15, 30), "noise_type": "compression", "reverb": 0.0},
        "CAR": {"snr": (12, 22), "noise_type": "engine", "reverb": 0.15},
    }
    
    def __init__(
        self,
        n_mels: int = 128,
        n_mfcc: int = 40,
        n_frames: int = 300,  # ~3 seconds at hop_length=512, sr=16000
        seed: int = 42,
    ):
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_frames = n_frames
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self._sample_counter = 0
    
    def _generate_mel_spectrogram(
        self,
        is_deepfake: bool,
        environment: str,
    ) -> torch.Tensor:
        """Generate a synthetic mel-spectrogram.
        
        Deepfake spectrograms have:
          - Smoother temporal variation
          - Artifacts in high-frequency bands (>6kHz)
          - Less natural harmonic structure
        """
        mel = np.zeros((self.n_mels, self.n_frames), dtype=np.float32)
        
        # Base harmonic structure (formants)
        num_formants = self.rng.randint(3, 6)
        for _ in range(num_formants):
            center_bin = self.rng.randint(5, self.n_mels - 10)
            bandwidth = self.rng.randint(3, 12)
            amplitude = self.np_rng.uniform(0.3, 1.0)
            
            for b in range(max(0, center_bin - bandwidth), min(self.n_mels, center_bin + bandwidth)):
                dist = abs(b - center_bin) / bandwidth
                formant_energy = amplitude * np.exp(-2 * dist**2)
                
                if is_deepfake:
                    # Smoother temporal variation for deepfakes
                    temporal = np.ones(self.n_frames) * formant_energy
                    temporal += self.np_rng.normal(0, 0.02, self.n_frames)
                else:
                    # Natural temporal variation
                    temporal = formant_energy * (1 + 0.3 * np.sin(
                        2 * np.pi * self.np_rng.uniform(0.5, 3.0) * np.arange(self.n_frames) / self.n_frames
                    ))
                    temporal += self.np_rng.normal(0, 0.08, self.n_frames)
                
                mel[b] += temporal
        
        # Deepfake artifacts: unnatural energy in high-frequency bands
        if is_deepfake:
            artifact_start = int(self.n_mels * 0.75)
            artifact_energy = self.np_rng.uniform(0.05, 0.15)
            mel[artifact_start:] += artifact_energy
            # Periodic artifacts from vocoder
            for f in range(artifact_start, self.n_mels):
                mel[f] += artifact_energy * np.sin(
                    2 * np.pi * 0.1 * np.arange(self.n_frames)
                )
        
        # Environment noise
        env_profile = self.ENVIRONMENT_PROFILES.get(environment, self.ENVIRONMENT_PROFILES["QUIET_ROOM"])
        snr = self.np_rng.uniform(*env_profile["snr"])
        noise_power = 10 ** (-snr / 20)
        mel += self.np_rng.normal(0, noise_power, mel.shape)
        
        # Reverb simulation
        reverb = env_profile["reverb"]
        if reverb > 0:
            kernel_size = max(1, int(reverb * 20))
            kernel = np.exp(-np.arange(kernel_size) / (reverb * 10))
            kernel /= kernel.sum()
            for b in range(self.n_mels):
                mel[b] = np.convolve(mel[b], kernel, mode="same")
        
        # Convert to log scale (like real mel-spectrograms)
        mel = np.maximum(mel, 1e-7)
        mel = np.log(mel)
        
        # Normalize to [-1, 1]
        mel = (mel - mel.mean()) / (mel.std() + 1e-7)
        
        return torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, n_frames)
    
    def _generate_mfcc(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """Derive MFCC-like features from the mel-spectrogram via DCT approximation."""
        mel_np = mel_spectrogram.squeeze(0).numpy()
        
        # Simplified DCT (using cosine basis)
        n_mfcc = min(self.n_mfcc, mel_np.shape[0])
        dct_matrix = np.zeros((n_mfcc, mel_np.shape[0]))
        for k in range(n_mfcc):
            for n in range(mel_np.shape[0]):
                dct_matrix[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2 * mel_np.shape[0]))
        
        mfcc = dct_matrix @ mel_np
        
        # Normalize
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-7)
        
        return torch.tensor(mfcc, dtype=torch.float32)
    
    def _generate_prosody(
        self,
        is_deepfake: bool,
        gender: Optional[str] = None,
    ) -> Dict[str, float]:
        """Generate realistic prosody features."""
        if is_deepfake:
            profile = self.PROSODY_PROFILES["deepfake"]
        else:
            if gender is None:
                gender = self.rng.choice(["male", "female"])
            profile = self.PROSODY_PROFILES[f"real_{gender}"]
        
        features = {}
        for feature_name, (low, high) in profile.items():
            features[feature_name] = self.np_rng.uniform(low, high)
        
        # Add derived features
        features["pitch_range"] = features["pitch_std"] * 4  # Approximate range
        features["energy_mean"] = self.np_rng.uniform(0.3, 0.8)
        features["energy_std"] = self.np_rng.uniform(0.05, 0.2) if not is_deepfake else self.np_rng.uniform(0.02, 0.08)
        features["zero_crossing_rate"] = self.np_rng.uniform(0.03, 0.12)
        features["spectral_centroid"] = self.np_rng.uniform(1500, 4000)
        
        return features
    
    def generate_sample(
        self,
        is_deepfake: Optional[bool] = None,
        environment: Optional[str] = None,
        gender: Optional[str] = None,
    ) -> GeneratedAudioFeatures:
        """Generate one complete audio feature sample.
        
        Args:
            is_deepfake: Whether the voice is AI-generated. If None, random (30% deepfake).
            environment: Call environment. If None, random.
            gender: Speaker gender. If None, random.
        """
        self._sample_counter += 1
        sample_id = f"audio_{self._sample_counter:05d}"
        
        if is_deepfake is None:
            is_deepfake = self.rng.random() < 0.30
        
        if environment is None:
            if is_deepfake:
                # Deepfakes more likely from VOIP/call centers
                environment = self.rng.choices(
                    ["VOIP", "CALL_CENTER", "QUIET_ROOM"],
                    weights=[0.5, 0.35, 0.15],
                    k=1,
                )[0]
            else:
                environment = self.rng.choice(list(self.ENVIRONMENT_PROFILES.keys()))
        
        mel = self._generate_mel_spectrogram(is_deepfake, environment)
        mfcc = self._generate_mfcc(mel)
        prosody = self._generate_prosody(is_deepfake, gender)
        
        return GeneratedAudioFeatures(
            sample_id=sample_id,
            mel_spectrogram=mel,
            mfcc=mfcc,
            prosody_features=prosody,
            is_deepfake=is_deepfake,
            environment=environment,
            labels={
                "deepfake_label": 1 if is_deepfake else 0,
                "environment_label": environment,
            },
        )
    
    def generate_batch(
        self,
        num_real: int = 500,
        num_deepfake: int = 200,
    ) -> List[GeneratedAudioFeatures]:
        """Generate a batch of audio feature samples."""
        samples = []
        
        for _ in range(num_real):
            samples.append(self.generate_sample(is_deepfake=False))
        
        for _ in range(num_deepfake):
            samples.append(self.generate_sample(is_deepfake=True))
        
        self.rng.shuffle(samples)
        return samples
