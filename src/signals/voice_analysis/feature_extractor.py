"""
Voice Feature Extractor — Low-level audio feature extraction.

Extracts acoustic features that distinguish:
  - Real human speech from AI-generated/deepfake voices
  - Stressed/confident speakers from scripted scammers
  - Different acoustic environments (call center vs. office vs. outdoor)

Features extracted:
  - MFCCs (40 coefficients) — spectral envelope
  - Pitch (F0) contour — fundamental frequency over time
  - Jitter — cycle-to-cycle pitch variation
  - Shimmer — cycle-to-cycle amplitude variation
  - HNR — harmonics-to-noise ratio
  - Speech rate — syllables per second
  - Pause patterns — duration and frequency of silences
  - Spectral features — centroid, bandwidth, rolloff
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple


class VoiceFeatureExtractor:
    """Extract acoustic features from audio waveforms.
    
    Uses numpy-based implementations to avoid heavy librosa dependency
    in the prototype. Production version would use librosa + torchaudio.
    """
    
    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 40,
                 n_fft: int = 2048, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract_all_features(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all acoustic features from a waveform.
        
        Args:
            waveform: 1D numpy array of audio samples (mono, normalized to [-1, 1])
            
        Returns:
            Dictionary mapping feature names to numpy arrays
        """
        features = {}
        
        # Basic spectral features
        features["mfcc"] = self._extract_mfcc(waveform)
        features["spectral_centroid"] = self._spectral_centroid(waveform)
        features["spectral_bandwidth"] = self._spectral_bandwidth(waveform)
        features["spectral_rolloff"] = self._spectral_rolloff(waveform)
        features["zero_crossing_rate"] = self._zero_crossing_rate(waveform)
        
        # Pitch and voice quality
        features["pitch_contour"] = self._estimate_pitch(waveform)
        features["jitter"] = self._compute_jitter(features["pitch_contour"])
        features["shimmer"] = self._compute_shimmer(waveform)
        features["hnr"] = self._compute_hnr(waveform)
        
        # Temporal features
        features["speech_rate"] = self._estimate_speech_rate(waveform)
        features["pause_features"] = self._analyze_pauses(waveform)
        features["energy_contour"] = self._energy_contour(waveform)
        
        # Mel spectrogram (for CNN input)
        features["mel_spectrogram"] = self._mel_spectrogram(waveform)
        
        return features
    
    def features_to_tensor(self, features: Dict[str, np.ndarray]) -> torch.Tensor:
        """Flatten all features into a single tensor for model input.
        
        Returns a fixed-size feature vector regardless of audio duration.
        """
        flat_features = []
        
        # MFCCs: take mean and std across time
        if "mfcc" in features and features["mfcc"].size > 0:
            mfcc = features["mfcc"]
            flat_features.extend(np.mean(mfcc, axis=1).tolist())
            flat_features.extend(np.std(mfcc, axis=1).tolist())
        else:
            flat_features.extend([0.0] * (self.n_mfcc * 2))
        
        # Spectral features: mean and std
        for key in ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "zero_crossing_rate"]:
            arr = features.get(key, np.array([0.0]))
            flat_features.append(float(np.mean(arr)))
            flat_features.append(float(np.std(arr)))
        
        # Scalar features
        flat_features.append(float(features.get("jitter", 0.0)))
        flat_features.append(float(features.get("shimmer", 0.0)))
        flat_features.append(float(features.get("hnr", 0.0)))
        flat_features.append(float(features.get("speech_rate", 0.0)))
        
        # Pause features
        pause = features.get("pause_features", {})
        flat_features.append(float(pause.get("num_pauses", 0)))
        flat_features.append(float(pause.get("mean_pause_duration", 0.0)))
        flat_features.append(float(pause.get("pause_ratio", 0.0)))
        
        # Pitch contour: mean, std, range
        pitch = features.get("pitch_contour", np.array([0.0]))
        voiced = pitch[pitch > 0]
        if len(voiced) > 0:
            flat_features.extend([
                float(np.mean(voiced)), float(np.std(voiced)),
                float(np.max(voiced) - np.min(voiced)),
            ])
        else:
            flat_features.extend([0.0, 0.0, 0.0])
        
        return torch.tensor(flat_features, dtype=torch.float32)
    
    def _extract_mfcc(self, waveform: np.ndarray) -> np.ndarray:
        """Extract MFCCs using DCT of log mel spectrogram."""
        mel_spec = self._mel_spectrogram(waveform)
        log_mel = np.log(mel_spec + 1e-10)
        
        # DCT-II to get MFCCs
        n_filters = log_mel.shape[0]
        n_frames = log_mel.shape[1]
        mfcc = np.zeros((self.n_mfcc, n_frames))
        
        for k in range(self.n_mfcc):
            for n in range(n_filters):
                mfcc[k] += log_mel[n] * np.cos(np.pi * k * (2 * n + 1) / (2 * n_filters))
        
        return mfcc
    
    def _mel_spectrogram(self, waveform: np.ndarray, n_mels: int = 128) -> np.ndarray:
        """Compute mel spectrogram."""
        # STFT
        num_frames = 1 + (len(waveform) - self.n_fft) // self.hop_length
        if num_frames <= 0:
            return np.zeros((n_mels, 1))
        
        frames = np.zeros((self.n_fft, num_frames))
        window = np.hanning(self.n_fft)
        
        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            if end <= len(waveform):
                frames[:, i] = waveform[start:end] * window
        
        # FFT
        spec = np.abs(np.fft.rfft(frames, axis=0))
        
        # Mel filterbank
        mel_filters = self._mel_filterbank(n_mels, spec.shape[0])
        mel_spec = mel_filters @ spec
        
        return mel_spec
    
    def _mel_filterbank(self, n_mels: int, n_fft_bins: int) -> np.ndarray:
        """Create mel filterbank matrix."""
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (self.sample_rate / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        bin_points = np.clip(bin_points, 0, n_fft_bins - 1)
        
        filters = np.zeros((n_mels, n_fft_bins))
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            for j in range(left, center):
                if center > left:
                    filters[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                if right > center:
                    filters[i, j] = (right - j) / (right - center)
        
        return filters
    
    def _spectral_centroid(self, waveform: np.ndarray) -> np.ndarray:
        """Compute spectral centroid over time."""
        mel = self._mel_spectrogram(waveform, n_mels=64)
        freqs = np.arange(mel.shape[0])
        centroid = np.sum(freqs[:, None] * mel, axis=0) / (np.sum(mel, axis=0) + 1e-10)
        return centroid
    
    def _spectral_bandwidth(self, waveform: np.ndarray) -> np.ndarray:
        """Compute spectral bandwidth over time."""
        mel = self._mel_spectrogram(waveform, n_mels=64)
        centroid = self._spectral_centroid(waveform)
        freqs = np.arange(mel.shape[0])
        diff = (freqs[:, None] - centroid[None, :]) ** 2
        bandwidth = np.sqrt(np.sum(diff * mel, axis=0) / (np.sum(mel, axis=0) + 1e-10))
        return bandwidth
    
    def _spectral_rolloff(self, waveform: np.ndarray, threshold: float = 0.85) -> np.ndarray:
        """Compute spectral rolloff point."""
        mel = self._mel_spectrogram(waveform, n_mels=64)
        total_energy = np.cumsum(mel, axis=0)
        rolloff_threshold = threshold * total_energy[-1]
        rolloff = np.argmax(total_energy >= rolloff_threshold[None, :], axis=0)
        return rolloff.astype(float)
    
    def _zero_crossing_rate(self, waveform: np.ndarray) -> np.ndarray:
        """Compute zero-crossing rate in frames."""
        num_frames = max(1, len(waveform) // self.hop_length)
        zcr = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * self.hop_length
            end = min(start + self.n_fft, len(waveform))
            frame = waveform[start:end]
            if len(frame) > 1:
                zcr[i] = np.sum(np.abs(np.diff(np.sign(frame)))) / (2.0 * len(frame))
        return zcr
    
    def _estimate_pitch(self, waveform: np.ndarray) -> np.ndarray:
        """Estimate pitch (F0) contour using autocorrelation method."""
        frame_size = self.n_fft
        num_frames = max(1, (len(waveform) - frame_size) // self.hop_length)
        pitch = np.zeros(num_frames)
        
        min_lag = int(self.sample_rate / 500)   # 500 Hz max pitch
        max_lag = int(self.sample_rate / 50)    # 50 Hz min pitch
        
        for i in range(num_frames):
            start = i * self.hop_length
            frame = waveform[start:start + frame_size]
            if len(frame) < frame_size:
                break
            
            # Autocorrelation
            frame = frame - np.mean(frame)
            acf = np.correlate(frame, frame, mode='full')
            acf = acf[len(acf) // 2:]
            
            if max_lag < len(acf) and min_lag < max_lag:
                search_region = acf[min_lag:max_lag]
                if len(search_region) > 0:
                    peak_idx = np.argmax(search_region) + min_lag
                    if acf[0] > 0 and acf[peak_idx] / acf[0] > 0.2:
                        pitch[i] = self.sample_rate / peak_idx
        
        return pitch
    
    def _compute_jitter(self, pitch_contour: np.ndarray) -> float:
        """Compute jitter (pitch perturbation) — marker for voice naturalness."""
        voiced = pitch_contour[pitch_contour > 0]
        if len(voiced) < 2:
            return 0.0
        periods = 1.0 / voiced
        jitter = np.mean(np.abs(np.diff(periods))) / np.mean(periods)
        return float(jitter)
    
    def _compute_shimmer(self, waveform: np.ndarray) -> float:
        """Compute shimmer (amplitude perturbation)."""
        frame_size = self.hop_length
        num_frames = len(waveform) // frame_size
        if num_frames < 2:
            return 0.0
        amplitudes = np.array([
            np.max(np.abs(waveform[i * frame_size:(i + 1) * frame_size]))
            for i in range(num_frames)
        ])
        amplitudes = amplitudes[amplitudes > 0]
        if len(amplitudes) < 2:
            return 0.0
        shimmer = np.mean(np.abs(np.diff(amplitudes))) / np.mean(amplitudes)
        return float(shimmer)
    
    def _compute_hnr(self, waveform: np.ndarray) -> float:
        """Compute harmonics-to-noise ratio — synthetic voices often have abnormal HNR."""
        frame = waveform[:self.n_fft]
        if len(frame) < self.n_fft:
            frame = np.pad(frame, (0, self.n_fft - len(frame)))
        
        frame = frame - np.mean(frame)
        acf = np.correlate(frame, frame, mode='full')
        acf = acf[len(acf) // 2:]
        
        if acf[0] <= 0:
            return 0.0
        
        # Find first peak in autocorrelation after zero crossing
        peak = np.max(acf[1:]) if len(acf) > 1 else 0
        if peak <= 0 or acf[0] <= peak:
            return 0.0
        
        hnr = 10 * np.log10(peak / (acf[0] - peak + 1e-10))
        return float(np.clip(hnr, -20, 40))
    
    def _estimate_speech_rate(self, waveform: np.ndarray) -> float:
        """Estimate speech rate (syllables per second)."""
        # Use energy contour to find syllable nuclei
        energy = self._energy_contour(waveform)
        if len(energy) < 3:
            return 0.0
        
        # Smooth energy
        kernel = np.ones(5) / 5
        smoothed = np.convolve(energy, kernel, mode='same')
        
        # Count peaks (syllable nuclei)
        threshold = np.mean(smoothed) * 0.5
        peaks = 0
        above = False
        for val in smoothed:
            if val > threshold and not above:
                peaks += 1
                above = True
            elif val <= threshold:
                above = False
        
        duration = len(waveform) / self.sample_rate
        return peaks / duration if duration > 0 else 0.0
    
    def _analyze_pauses(self, waveform: np.ndarray, threshold: float = 0.02) -> Dict:
        """Analyze pause patterns — scripted scammers have different pause patterns."""
        energy = self._energy_contour(waveform)
        
        is_silence = energy < threshold
        pauses = []
        current_pause_len = 0
        
        for silent in is_silence:
            if silent:
                current_pause_len += 1
            else:
                if current_pause_len > 2:  # Min pause length
                    pauses.append(current_pause_len * self.hop_length / self.sample_rate)
                current_pause_len = 0
        
        if current_pause_len > 2:
            pauses.append(current_pause_len * self.hop_length / self.sample_rate)
        
        duration = len(waveform) / self.sample_rate
        return {
            "num_pauses": len(pauses),
            "mean_pause_duration": float(np.mean(pauses)) if pauses else 0.0,
            "max_pause_duration": float(np.max(pauses)) if pauses else 0.0,
            "pause_ratio": float(np.sum(pauses) / duration) if duration > 0 else 0.0,
        }
    
    def _energy_contour(self, waveform: np.ndarray) -> np.ndarray:
        """Compute frame-level energy contour."""
        num_frames = max(1, len(waveform) // self.hop_length)
        energy = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * self.hop_length
            end = min(start + self.hop_length, len(waveform))
            frame = waveform[start:end]
            energy[i] = np.sqrt(np.mean(frame ** 2))
        return energy
