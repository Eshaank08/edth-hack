#!/usr/bin/env python3
"""
Elite feature extraction - maximum information extraction.
Includes chromagrams, spectral contrast, and more advanced features.
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


class EliteAudioFeatureExtractor:
    """
    Elite feature extraction with maximum information.
    Combines mel-spectrogram, MFCCs, chromagram, spectral contrast, and more.
    """
    
    def __init__(
        self,
        n_mels: int = 128,
        n_mfcc: int = 40,
        n_chroma: int = 12,
        max_duration: float = 3.0,
        sample_rate: int = 44100,
    ):
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_chroma = n_chroma
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_length = int(max_duration * sample_rate)
        
        # Scalers for different feature types
        self.mel_scaler = StandardScaler()
        self.mfcc_scaler = StandardScaler()
        self.chroma_scaler = StandardScaler()
        self.contrast_scaler = StandardScaler()
        self.fitted = False
    
    def _pad_or_crop(self, audio: np.ndarray) -> np.ndarray:
        """Pad or crop audio to fixed length."""
        if len(audio) > self.max_length:
            start = (len(audio) - self.max_length) // 2
            audio = audio[start:start + self.max_length]
        elif len(audio) < self.max_length:
            padding = self.max_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        return audio
    
    def extract_mel_spectrogram(self, waveform: AudioWaveform) -> np.ndarray:
        """Extract mel-spectrogram with higher resolution."""
        audio = waveform.data.numpy()
        audio = self._pad_or_crop(audio)
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=int(waveform.sample_rate),
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512,
            fmax=8000,
            window='hann',
        )
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_mfcc_temporal(self, waveform: AudioWaveform) -> np.ndarray:
        """Extract MFCCs with delta and delta-delta."""
        audio = waveform.data.numpy()
        audio = self._pad_or_crop(audio)
        
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=int(waveform.sample_rate),
            n_mfcc=self.n_mfcc,
            n_fft=2048,
            hop_length=512,
        )
        
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        return np.vstack([mfccs, mfcc_delta, mfcc_delta2])
    
    def extract_chromagram(self, waveform: AudioWaveform) -> np.ndarray:
        """
        Extract chromagram (pitch class profiles).
        Useful for detecting tonal patterns in drone sounds.
        """
        audio = waveform.data.numpy()
        audio = self._pad_or_crop(audio)
        
        chroma = librosa.feature.chroma_cqt(
            y=audio,
            sr=int(waveform.sample_rate),
            hop_length=512,
            n_chroma=self.n_chroma,
        )
        
        return chroma
    
    def extract_spectral_contrast(self, waveform: AudioWaveform) -> np.ndarray:
        """
        Extract spectral contrast (difference between peaks and valleys).
        Captures textural differences between drone/helicopter/background.
        """
        audio = waveform.data.numpy()
        audio = self._pad_or_crop(audio)
        
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=int(waveform.sample_rate),
            n_fft=2048,
            hop_length=512,
            n_bands=6,
        )
        
        return contrast
    
    def extract_statistical_features(self, waveform: AudioWaveform) -> np.ndarray:
        """Extract comprehensive statistical features."""
        audio = waveform.data.numpy()
        features = []
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.extend([np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr)])
        
        # Spectral centroid
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=int(waveform.sample_rate))[0]
        features.extend([np.mean(spec_cent), np.std(spec_cent), np.max(spec_cent)])
        
        # Spectral rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=int(waveform.sample_rate))[0]
        features.extend([np.mean(spec_rolloff), np.std(spec_rolloff)])
        
        # Spectral bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=int(waveform.sample_rate))[0]
        features.extend([np.mean(spec_bw), np.std(spec_bw)])
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        features.extend([np.mean(rms), np.std(rms), np.max(rms)])
        
        # Spectral flatness (measure of noisiness)
        flatness = librosa.feature.spectral_flatness(y=audio)[0]
        features.extend([np.mean(flatness), np.std(flatness)])
        
        # Tempo and beat
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=int(waveform.sample_rate))
            features.append(tempo)
        except:
            features.append(0.0)
        
        return np.array(features)
    
    def extract_features(self, waveform: AudioWaveform) -> dict[str, np.ndarray]:
        """Extract all features from a waveform."""
        return {
            'mel_spec': self.extract_mel_spectrogram(waveform),
            'mfcc': self.extract_mfcc_temporal(waveform),
            'chroma': self.extract_chromagram(waveform),
            'contrast': self.extract_spectral_contrast(waveform),
            'stats': self.extract_statistical_features(waveform),
        }
    
    def fit(self, waveforms: list[AudioWaveform]) -> None:
        """Fit scalers on training data."""
        logger.info("Fitting elite feature scalers...")
        
        all_mel_specs = []
        all_mfccs = []
        all_chromas = []
        all_contrasts = []
        
        for waveform in waveforms:
            features = self.extract_features(waveform)
            all_mel_specs.append(features['mel_spec'].flatten())
            all_mfccs.append(features['mfcc'].flatten())
            all_chromas.append(features['chroma'].flatten())
            all_contrasts.append(features['contrast'].flatten())
        
        self.mel_scaler.fit(np.array(all_mel_specs))
        self.mfcc_scaler.fit(np.array(all_mfccs))
        self.chroma_scaler.fit(np.array(all_chromas))
        self.contrast_scaler.fit(np.array(all_contrasts))
        self.fitted = True
        
        logger.info("Elite feature scalers fitted successfully")
    
    def transform(self, waveforms: list[AudioWaveform]) -> dict[str, torch.Tensor]:
        """Transform waveforms to normalized features."""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        batch_mel_specs = []
        batch_mfccs = []
        batch_chromas = []
        batch_contrasts = []
        batch_stats = []
        
        for waveform in waveforms:
            features = self.extract_features(waveform)
            
            # Normalize each feature type
            mel_flat = features['mel_spec'].flatten()
            mel_normalized = self.mel_scaler.transform(mel_flat.reshape(1, -1))[0]
            mel_normalized = mel_normalized.reshape(features['mel_spec'].shape)
            
            mfcc_flat = features['mfcc'].flatten()
            mfcc_normalized = self.mfcc_scaler.transform(mfcc_flat.reshape(1, -1))[0]
            mfcc_normalized = mfcc_normalized.reshape(features['mfcc'].shape)
            
            chroma_flat = features['chroma'].flatten()
            chroma_normalized = self.chroma_scaler.transform(chroma_flat.reshape(1, -1))[0]
            chroma_normalized = chroma_normalized.reshape(features['chroma'].shape)
            
            contrast_flat = features['contrast'].flatten()
            contrast_normalized = self.contrast_scaler.transform(contrast_flat.reshape(1, -1))[0]
            contrast_normalized = contrast_normalized.reshape(features['contrast'].shape)
            
            batch_mel_specs.append(mel_normalized)
            batch_mfccs.append(mfcc_normalized)
            batch_chromas.append(chroma_normalized)
            batch_contrasts.append(contrast_normalized)
            batch_stats.append(features['stats'])
        
        return {
            'mel_spec': torch.from_numpy(np.array(batch_mel_specs)).float(),
            'mfcc': torch.from_numpy(np.array(batch_mfccs)).float(),
            'chroma': torch.from_numpy(np.array(batch_chromas)).float(),
            'contrast': torch.from_numpy(np.array(batch_contrasts)).float(),
            'stats': torch.from_numpy(np.array(batch_stats)).float(),
        }

