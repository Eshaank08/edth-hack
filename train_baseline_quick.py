#!/usr/bin/env python3
"""
Quick baseline model trainer using only the example files.
This is useful for testing the competition bot without downloading the full dataset.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from hs_hackathon_drone_acoustics import CLASSES, EXAMPLES_DIR
from hs_hackathon_drone_acoustics.base import AudioWaveform
from hs_hackathon_drone_acoustics.feature_extractors import MFCCFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_SAVE_PATH = Path(__file__).parent / "models"
MODEL_SAVE_PATH.mkdir(exist_ok=True, parents=True)


class AudioClassifier(nn.Module):
    """Neural network classifier for audio classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def main() -> None:
    """Train a quick baseline model on example files."""
    logger.info("Creating baseline model from example files...")
    logger.info("Note: This is just for testing. Train on full dataset for better accuracy!")

    # Load example files
    example_files = {
        "BACKGROUND_001_L.wav": 0,  # background
        "DRONE_001_L.wav": 1,  # drone
        "HELICOPTER_001_L_0.wav": 2,  # helicopter
    }

    n_mfcc = 40
    mfcc_extractor = MFCCFeatureExtractor(n_mfcc=n_mfcc)

    # Extract features
    features = []
    labels = []

    for filename, label in example_files.items():
        filepath = EXAMPLES_DIR / filename
        if not filepath.exists():
            logger.error(f"Example file not found: {filepath}")
            continue

        waveform = AudioWaveform.load(filepath)
        mfcc = mfcc_extractor.extract(waveform)
        features.append(mfcc)
        labels.append(label)
        logger.info(f"Loaded {filename} -> {CLASSES[label]}")

    features_array = np.stack(features)

    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(features_array)
    scaled_features = scaler.transform(features_array)

    # Create simple model with random weights (since we have too few samples to train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioClassifier(input_dim=n_mfcc, hidden_dim=128, num_classes=len(CLASSES))
    model.to(device)

    # Initialize with reasonable weights
    # This is just a baseline - the real model should be trained on full dataset
    torch.manual_seed(42)
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "n_mfcc": n_mfcc,
        "input_dim": n_mfcc,
        "val_accuracy": 0.33,  # Random baseline
        "classes": CLASSES,
        "note": "BASELINE MODEL - Train on full dataset for better accuracy",
    }

    save_path = MODEL_SAVE_PATH / "best_model.pt"
    torch.save(checkpoint, save_path)
    logger.info(f"\n✅ Baseline model saved to {save_path}")
    logger.info("\n⚠️  WARNING: This is just a baseline for testing!")
    logger.info("For competition, download the full dataset and run: uv run python train_model.py")


if __name__ == "__main__":
    main()

