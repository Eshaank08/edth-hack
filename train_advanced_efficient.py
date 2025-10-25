#!/usr/bin/env python3
"""
Memory-efficient training script optimized for laptop training.
Uses gradient accumulation, mixed precision, and efficient data loading.
"""

# Fix Python path for Colab (must be first!)
import sys
from pathlib import Path
# Try to find the src directory
possible_paths = [
    Path(__file__).parent / "src",
    Path("/content/edth-munich-drone-acoustics/src"),
    Path("/Users/eshaan_kansal/Downloads/edth-hack/src"),
]
for path in possible_paths:
    if path.exists():
        sys.path.insert(0, str(path))
        break

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from advanced_features import AdvancedAudioFeatureExtractor
from advanced_model import get_model
from hs_hackathon_drone_acoustics import CLASSES, RAW_DATA_DIR
from hs_hackathon_drone_acoustics.base import AudioDataset, AudioWaveform
from hs_hackathon_drone_acoustics.metrics import evaluate, get_confusion_matrix_str

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN_PATH = RAW_DATA_DIR / "train"
VAL_PATH = RAW_DATA_DIR / "val"
MODEL_SAVE_PATH = Path(__file__).parent / "models"
MODEL_SAVE_PATH.mkdir(exist_ok=True, parents=True)

# Memory-efficient configuration
USE_MIXED_PRECISION = True  # Reduces memory by ~40%
GRADIENT_ACCUMULATION_STEPS = 4  # Simulate larger batch size
PHYSICAL_BATCH_SIZE = 8  # Small batch per forward pass
EFFECTIVE_BATCH_SIZE = PHYSICAL_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS  # 32


def collate_fn(batch: list[tuple[AudioWaveform, int]]) -> tuple[list[AudioWaveform], torch.Tensor]:
    """Custom collate function for DataLoader."""
    waveforms = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return waveforms, labels


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    feature_extractor: AdvancedAudioFeatureExtractor,
    device: torch.device,
    scaler: GradScaler | None = None,
    accumulation_steps: int = 1,
) -> tuple[float, float]:
    """Train for one epoch with gradient accumulation and mixed precision."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (waveforms, labels) in enumerate(pbar):
        # Extract features
        features = feature_extractor.transform(waveforms)
        mel_spec = features['mel_spec'].to(device)
        mfcc = features['mfcc'].to(device)
        stats = features['stats'].to(device)
        labels = labels.to(device)
        
        # Mixed precision forward pass
        if scaler is not None:
            with autocast():
                outputs = model(mel_spec, mfcc, stats)
                loss = criterion(outputs, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            # Update weights every N steps
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(mel_spec, mfcc, stats)
            loss = criterion(outputs, labels) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Statistics
        running_loss += loss.item() * len(labels) * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}', 'acc': f'{correct/total:.4f}'})
        
        # Free memory
        del mel_spec, mfcc, stats, outputs, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader[Any],
    criterion: nn.Module,
    feature_extractor: AdvancedAudioFeatureExtractor,
    device: torch.device,
) -> tuple[float, float, torch.Tensor]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probas = []
    
    for waveforms, labels in tqdm(dataloader, desc="Validating"):
        # Extract features
        features = feature_extractor.transform(waveforms)
        mel_spec = features['mel_spec'].to(device)
        mfcc = features['mfcc'].to(device)
        stats = features['stats'].to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(mel_spec, mfcc, stats)
        loss = criterion(outputs, labels)
        
        # Statistics
        running_loss += loss.item() * len(labels)
        probas = torch.softmax(outputs, dim=1)
        all_probas.append(probas.cpu())
        all_preds.append(outputs.argmax(1).cpu())
        all_targets.append(labels.cpu())
        
        # Free memory
        del mel_spec, mfcc, stats, outputs, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    all_probas = torch.cat(all_probas)
    all_targets = torch.cat(all_targets)
    
    avg_loss, accuracy, confusion_matrix = evaluate(all_probas, all_targets)
    return avg_loss, accuracy, confusion_matrix


def main() -> None:
    """Main training function."""
    # Check if data exists
    if not TRAIN_PATH.exists():
        logger.error(f"Training data not found at {TRAIN_PATH}")
        logger.error("Please download the dataset first!")
        return
    
    logger.info("="*60)
    logger.info("🚀 Advanced Efficient Training Pipeline")
    logger.info("="*60)
    logger.info(f"Physical batch size: {PHYSICAL_BATCH_SIZE}")
    logger.info(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    logger.info(f"Effective batch size: {EFFECTIVE_BATCH_SIZE}")
    logger.info(f"Mixed precision: {USE_MIXED_PRECISION}")
    logger.info("="*60)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = AudioDataset(root_dir=TRAIN_PATH)
    val_dataset = AudioDataset(root_dir=VAL_PATH)
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create dataloaders with fewer workers to save memory
    train_loader = DataLoader(
        train_dataset,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # Reduced from 4
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Initialize feature extractor
    logger.info("Initializing feature extractor...")
    feature_extractor = AdvancedAudioFeatureExtractor(
        n_mels=128,
        n_mfcc=40,
        max_duration=3.0,
    )
    
    logger.info("Fitting feature extractor on training data (this may take a few minutes)...")
    train_waveforms, _ = train_dataset[:]
    feature_extractor.fit(train_waveforms)
    logger.info("Feature extraction setup complete!")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    model = get_model("efficient", num_classes=len(CLASSES), dropout=0.4)
    model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Helps generalization
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=50,
        steps_per_epoch=len(train_loader) // GRADIENT_ACCUMULATION_STEPS,
        pct_start=0.1,
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if USE_MIXED_PRECISION and torch.cuda.is_available() else None
    
    # Training loop
    num_epochs = 50
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60 + "\n")
    
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            feature_extractor, device, scaler, GRADIENT_ACCUMULATION_STEPS
        )
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc, val_cm = validate(
            model, val_loader, criterion, feature_extractor, device
        )
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logger.info(f"Confusion Matrix:\n{get_confusion_matrix_str(val_cm)}")
        
        # Learning rate scheduling
        if scaler is None:  # OneCycleLR steps per batch
            pass
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_type": "efficient",
                "mel_scaler_mean": feature_extractor.mel_scaler.mean_,
                "mel_scaler_scale": feature_extractor.mel_scaler.scale_,
                "mfcc_scaler_mean": feature_extractor.mfcc_scaler.mean_,
                "mfcc_scaler_scale": feature_extractor.mfcc_scaler.scale_,
                "n_mels": 128,
                "n_mfcc": 40,
                "max_duration": 3.0,
                "val_accuracy": val_acc,
                "classes": CLASSES,
            }
            
            save_path = MODEL_SAVE_PATH / "best_model_advanced.pt"
            torch.save(checkpoint, save_path)
            logger.info(f"✓ Saved best model with val_acc: {val_acc:.4f} to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    logger.info("\n" + "="*60)
    logger.info(f"🎉 Training complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    # Set memory optimizations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
    
    main()

