#!/usr/bin/env python3
"""
Train a robust audio classification model for drone detection.
This script trains a neural network classifier on the provided dataset.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from hs_hackathon_drone_acoustics import CLASSES, RAW_DATA_DIR
from hs_hackathon_drone_acoustics.base import AudioDataset, AudioWaveform
from hs_hackathon_drone_acoustics.feature_extractors import MFCCFeatureExtractor
from hs_hackathon_drone_acoustics.metrics import evaluate, get_confusion_matrix_str

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN_PATH = RAW_DATA_DIR / "train"
VAL_PATH = RAW_DATA_DIR / "val"
MODEL_SAVE_PATH = Path(__file__).parent / "models"
MODEL_SAVE_PATH.mkdir(exist_ok=True, parents=True)


class AudioFeatureExtractor:
    """Extract MFCC features from audio waveforms."""

    def __init__(self, n_mfcc: int = 40):
        self.mfcc_extractor = MFCCFeatureExtractor(n_mfcc=n_mfcc)
        self.scaler = StandardScaler()
        self.fitted = False

    def extract_features(self, waveforms: list[AudioWaveform]) -> np.ndarray:
        """Extract MFCC features from a list of waveforms."""
        features = []
        for waveform in waveforms:
            mfcc = self.mfcc_extractor.extract(waveform)
            features.append(mfcc)
        return np.stack(features)

    def fit(self, waveforms: list[AudioWaveform]) -> None:
        """Fit the scaler on training data."""
        features = self.extract_features(waveforms)
        self.scaler.fit(features)
        self.fitted = True

    def transform(self, waveforms: list[AudioWaveform]) -> torch.Tensor:
        """Transform waveforms to normalized features."""
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")
        features = self.extract_features(waveforms)
        scaled_features = self.scaler.transform(features)
        return torch.from_numpy(scaled_features).float()


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


def collate_fn(batch: list[tuple[AudioWaveform, int]]) -> tuple[list[AudioWaveform], torch.Tensor]:
    """Custom collate function for DataLoader."""
    waveforms = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return waveforms, labels


def train_epoch(
    model: AudioClassifier,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    feature_extractor: AudioFeatureExtractor,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for waveforms, labels in tqdm(dataloader, desc="Training"):
        # Extract features
        features = feature_extractor.transform(waveforms).to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * len(labels)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(
    model: AudioClassifier,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    feature_extractor: AudioFeatureExtractor,
    device: torch.device,
) -> tuple[float, float, torch.Tensor]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probas = []

    with torch.no_grad():
        for waveforms, labels in tqdm(dataloader, desc="Validating"):
            # Extract features
            features = feature_extractor.transform(waveforms).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * len(labels)
            probas = torch.softmax(outputs, dim=1)
            all_probas.append(probas.cpu())
            all_preds.append(outputs.argmax(1).cpu())
            all_targets.append(labels.cpu())

    all_probas = torch.cat(all_probas)
    all_targets = torch.cat(all_targets)

    avg_loss, accuracy, confusion_matrix = evaluate(all_probas, all_targets)
    return avg_loss, accuracy, confusion_matrix


def main() -> None:
    """Main training function."""
    # Check if data exists
    if not TRAIN_PATH.exists():
        logger.error(f"Training data not found at {TRAIN_PATH}")
        logger.error("Please download the dataset from:")
        logger.error(
            "https://github.com/helsing-ai/edth-munich-drone-acoustics/releases/download/train_val_data/drone_acoustics_train_val_data.zip"
        )
        logger.error("Extract and place train/ and val/ folders in data/raw/")
        return

    logger.info("Loading datasets...")
    train_dataset = AudioDataset(root_dir=TRAIN_PATH)
    val_dataset = AudioDataset(root_dir=VAL_PATH)
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Initialize feature extractor
    logger.info("Fitting feature extractor on training data...")
    feature_extractor = AudioFeatureExtractor(n_mfcc=40)
    train_waveforms, _ = train_dataset[:]
    feature_extractor.fit(train_waveforms)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    input_dim = 40  # n_mfcc
    model = AudioClassifier(input_dim=input_dim, hidden_dim=128, num_classes=len(CLASSES)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Training loop
    num_epochs = 50
    best_val_acc = 0.0

    logger.info("Starting training...")
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, feature_extractor, device)
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        val_loss, val_acc, val_cm = validate(model, val_loader, criterion, feature_extractor, device)
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.info(f"Confusion Matrix:\n{get_confusion_matrix_str(val_cm)}")

        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "scaler_mean": feature_extractor.scaler.mean_,
                "scaler_scale": feature_extractor.scaler.scale_,
                "n_mfcc": 40,
                "input_dim": input_dim,
                "val_accuracy": val_acc,
                "classes": CLASSES,
            }
            save_path = MODEL_SAVE_PATH / "best_model.pt"
            torch.save(checkpoint, save_path)
            logger.info(f"Saved best model with val_acc: {val_acc:.4f} to {save_path}")

    logger.info(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()

