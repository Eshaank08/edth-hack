#!/usr/bin/env python3
"""
Advanced feature extraction for audio classification.
Includes spectrograms, mel-spectrograms, and temporal features.
"""

# Fix Python path for Colab
import sys
from pathlib import Path
possible_paths = [
    Path(__file__).parent / "src",
    Path("/content/edth-munich-drone-acoustics/src"),
    Path("/Users/eshaan_kansal/Downloads/edth-hack/src"),
]
for path in possible_paths:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))
        break

import logging
from typing import Any

import librosa
import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from hs_hackathon_drone_acoustics.base import AudioWaveform

logger = logging.getLogger(__name__)


class AdvancedAudioFeatureExtractor:
    """Extract advanced audio features including spectrograms and temporal MFCCs."""
    
    def __init__(
        self,
        n_mels: int = 128,
        n_mfcc: int = 40,
        max_duration: float = 3.0,
        sample_rate: int = 44100,
    ):
        """
        Initialize the advanced feature extractor.
        
        Args:
            n_mels: Number of mel bands for mel-spectrogram
            n_mfcc: Number of MFCC coefficients
            max_duration: Maximum duration to process (will pad/crop)
            sample_rate: Target sample rate
        """
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_length = int(max_duration * sample_rate)
        
        # Scalers for different feature types
        self.mel_scaler = StandardScaler()
        self.mfcc_scaler = StandardScaler()
        self.fitted = False
    
    def _pad_or_crop(self, audio: np.ndarray) -> np.ndarray:
        """Pad or crop audio to fixed length."""
        if len(audio) > self.max_length:
            # Crop from center
            start = (len(audio) - self.max_length) // 2
            audio = audio[start:start + self.max_length]
        elif len(audio) < self.max_length:
            # Pad with zeros
            padding = self.max_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        return audio
    
    def extract_mel_spectrogram(self, waveform: AudioWaveform) -> np.ndarray:
        """
        Extract mel-spectrogram features.
        
        Returns:
            Mel-spectrogram of shape (n_mels, time_steps)
        """
        audio = waveform.data.numpy()
        audio = self._pad_or_crop(audio)
        
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=int(waveform.sample_rate),
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512,
            fmax=8000,  # Focus on relevant frequency range
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_mfcc_temporal(self, waveform: AudioWaveform) -> np.ndarray:
        """
        Extract temporal MFCC features with delta and delta-delta.
        
        Returns:
            MFCCs of shape (n_mfcc * 3, time_steps) - includes delta and delta-delta
        """
        audio = waveform.data.numpy()
        audio = self._pad_or_crop(audio)
        
        # Compute MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=int(waveform.sample_rate),
            n_mfcc=self.n_mfcc,
            n_fft=2048,
            hop_length=512,
        )
        
        # Compute delta and delta-delta (velocity and acceleration)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # Stack all features
        mfcc_features = np.vstack([mfccs, mfcc_delta, mfcc_delta2])
        
        return mfcc_features
    
    def extract_statistical_features(self, waveform: AudioWaveform) -> np.ndarray:
        """Extract statistical features from audio."""
        audio = waveform.data.numpy()
        
        features = []
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.extend([np.mean(zcr), np.std(zcr), np.max(zcr)])
        
        # Spectral centroid
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=int(waveform.sample_rate))[0]
        features.extend([np.mean(spec_cent), np.std(spec_cent)])
        
        # Spectral rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=int(waveform.sample_rate))[0]
        features.extend([np.mean(spec_rolloff), np.std(spec_rolloff)])
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        features.extend([np.mean(rms), np.std(rms), np.max(rms)])
        
        return np.array(features)
    
    def extract_features(self, waveform: AudioWaveform) -> dict[str, np.ndarray]:
        """
        Extract all features from a waveform.
        
        Returns:
            Dictionary with 'mel_spec', 'mfcc', and 'stats' keys
        """
        mel_spec = self.extract_mel_spectrogram(waveform)
        mfcc = self.extract_mfcc_temporal(waveform)
        stats = self.extract_statistical_features(waveform)
        
        return {
            'mel_spec': mel_spec,  # (n_mels, time_steps)
            'mfcc': mfcc,          # (n_mfcc * 3, time_steps)
            'stats': stats,        # (n_stats,)
        }
    
    def fit(self, waveforms: list[AudioWaveform]) -> None:
        """Fit scalers on training data."""
        logger.info("Fitting feature scalers...")
        
        all_mel_specs = []
        all_mfccs = []
        
        for waveform in waveforms:
            features = self.extract_features(waveform)
            all_mel_specs.append(features['mel_spec'].flatten())
            all_mfccs.append(features['mfcc'].flatten())
        
        # Fit scalers
        self.mel_scaler.fit(np.array(all_mel_specs))
        self.mfcc_scaler.fit(np.array(all_mfccs))
        self.fitted = True
        
        logger.info("Feature scalers fitted successfully")
    
    def transform(self, waveforms: list[AudioWaveform]) -> dict[str, torch.Tensor]:
        """
        Transform waveforms to normalized features.
        
        Returns:
            Dictionary with tensors ready for model input
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        batch_mel_specs = []
        batch_mfccs = []
        batch_stats = []
        
        for waveform in waveforms:
            features = self.extract_features(waveform)
            
            # Normalize mel-spectrogram
            mel_flat = features['mel_spec'].flatten()
            mel_normalized = self.mel_scaler.transform(mel_flat.reshape(1, -1))[0]
            mel_normalized = mel_normalized.reshape(features['mel_spec'].shape)
            
            # Normalize MFCCs
            mfcc_flat = features['mfcc'].flatten()
            mfcc_normalized = self.mfcc_scaler.transform(mfcc_flat.reshape(1, -1))[0]
            mfcc_normalized = mfcc_normalized.reshape(features['mfcc'].shape)
            
            batch_mel_specs.append(mel_normalized)
            batch_mfccs.append(mfcc_normalized)
            batch_stats.append(features['stats'])
        
        return {
            'mel_spec': torch.from_numpy(np.array(batch_mel_specs)).float(),
            'mfcc': torch.from_numpy(np.array(batch_mfccs)).float(),
            'stats': torch.from_numpy(np.array(batch_stats)).float(),
        }

