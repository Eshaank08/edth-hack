#!/usr/bin/env python3
"""
Data augmentation for audio training.
Increases dataset diversity and model robustness.
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

import numpy as np
import librosa
import torch
from hs_hackathon_drone_acoustics.base import AudioWaveform


class AudioAugmenter:
    """Apply various augmentations to audio waveforms."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
    
    def time_stretch(self, waveform: AudioWaveform, rate: float = 1.0) -> AudioWaveform:
        """
        Stretch or compress audio in time without changing pitch.
        rate > 1.0: faster, rate < 1.0: slower
        """
        audio = waveform.data.numpy()
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        return AudioWaveform(torch.from_numpy(stretched).float(), waveform.sample_rate)
    
    def pitch_shift(self, waveform: AudioWaveform, n_steps: float = 0.0) -> AudioWaveform:
        """
        Shift pitch without changing tempo.
        n_steps: number of semitones to shift (can be fractional)
        """
        audio = waveform.data.numpy()
        shifted = librosa.effects.pitch_shift(
            audio,
            sr=int(waveform.sample_rate),
            n_steps=n_steps
        )
        return AudioWaveform(torch.from_numpy(shifted).float(), waveform.sample_rate)
    
    def add_noise(self, waveform: AudioWaveform, noise_factor: float = 0.005) -> AudioWaveform:
        """Add white noise to audio."""
        audio = waveform.data.numpy()
        noise = np.random.normal(0, noise_factor, audio.shape)
        noisy_audio = audio + noise
        return AudioWaveform(torch.from_numpy(noisy_audio).float(), waveform.sample_rate)
    
    def change_volume(self, waveform: AudioWaveform, gain_db: float = 0.0) -> AudioWaveform:
        """Change volume by gain_db decibels."""
        audio = waveform.data.numpy()
        gain_factor = 10 ** (gain_db / 20.0)
        amplified = audio * gain_factor
        # Clip to prevent distortion
        amplified = np.clip(amplified, -1.0, 1.0)
        return AudioWaveform(torch.from_numpy(amplified).float(), waveform.sample_rate)
    
    def random_crop(self, waveform: AudioWaveform, crop_ratio: float = 0.9) -> AudioWaveform:
        """Randomly crop a portion of the audio."""
        audio = waveform.data.numpy()
        crop_length = int(len(audio) * crop_ratio)
        if crop_length >= len(audio):
            return waveform
        start = np.random.randint(0, len(audio) - crop_length)
        cropped = audio[start:start + crop_length]
        # Pad back to original length
        padding = len(audio) - len(cropped)
        cropped = np.pad(cropped, (0, padding), mode='constant')
        return AudioWaveform(torch.from_numpy(cropped).float(), waveform.sample_rate)
    
    def augment_random(self, waveform: AudioWaveform, augmentation_prob: float = 0.5) -> AudioWaveform:
        """
        Apply random augmentations with given probability.
        Used during training to increase data diversity.
        """
        augmented = waveform
        
        # Time stretch (0.9x - 1.1x)
        if np.random.random() < augmentation_prob:
            rate = np.random.uniform(0.9, 1.1)
            augmented = self.time_stretch(augmented, rate)
        
        # Pitch shift (±2 semitones)
        if np.random.random() < augmentation_prob:
            n_steps = np.random.uniform(-2, 2)
            augmented = self.pitch_shift(augmented, n_steps)
        
        # Add noise
        if np.random.random() < augmentation_prob:
            noise_factor = np.random.uniform(0.002, 0.008)
            augmented = self.add_noise(augmented, noise_factor)
        
        # Volume change (±6 dB)
        if np.random.random() < augmentation_prob:
            gain_db = np.random.uniform(-6, 6)
            augmented = self.change_volume(augmented, gain_db)
        
        return augmented
    
    def augment_tta(self, waveform: AudioWaveform) -> list[AudioWaveform]:
        """
        Test-time augmentation (TTA): create multiple augmented versions.
        Use for ensemble predictions at inference time.
        """
        versions = [waveform]  # Original
        
        # Slight time stretches
        versions.append(self.time_stretch(waveform, 0.95))
        versions.append(self.time_stretch(waveform, 1.05))
        
        # Slight pitch shifts
        versions.append(self.pitch_shift(waveform, -1.0))
        versions.append(self.pitch_shift(waveform, 1.0))
        
        return versions

