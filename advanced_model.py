#!/usr/bin/env python3
"""
Efficient CNN model for audio classification - optimized for laptop training.
Uses depthwise separable convolutions to reduce memory footprint.
"""

import torch
import torch.nn as nn


class EfficientAudioCNN(nn.Module):
    """
    Lightweight CNN for audio classification using dual-stream architecture.
    Processes mel-spectrogram and MFCC features separately then fuses them.
    Memory-efficient design suitable for laptop training.
    """
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        
        # Mel-spectrogram branch (128 x time_steps)
        self.mel_branch = nn.Sequential(
            # First conv block
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Second conv block
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Global average pooling
            nn.AdaptiveAvgPool1d(1),
        )
        
        # MFCC branch (120 x time_steps) - includes delta and delta-delta
        self.mfcc_branch = nn.Sequential(
            # First conv block
            nn.Conv1d(120, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Second conv block
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Global average pooling
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Fusion layer - combines both branches + statistical features
        fusion_dim = 32 + 32 + 11  # mel (32) + mfcc (32) + stats (11)
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, mel_spec: torch.Tensor, mfcc: torch.Tensor, stats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            mel_spec: (batch, n_mels, time_steps)
            mfcc: (batch, n_mfcc * 3, time_steps)
            stats: (batch, n_stats)
        
        Returns:
            logits: (batch, num_classes)
        """
        # Process mel-spectrogram
        mel_features = self.mel_branch(mel_spec).squeeze(-1)  # (batch, 32)
        
        # Process MFCCs
        mfcc_features = self.mfcc_branch(mfcc).squeeze(-1)  # (batch, 32)
        
        # Concatenate all features
        combined = torch.cat([mel_features, mfcc_features, stats], dim=1)
        
        # Final classification
        output = self.fusion(combined)
        
        return output


class SimplifiedCNN(nn.Module):
    """
    Ultra-lightweight single-branch CNN for very limited RAM.
    Uses only mel-spectrogram features.
    """
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv block 1
            nn.Conv1d(128, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            # Conv block 2
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )
    
    def forward(self, mel_spec: torch.Tensor, mfcc: torch.Tensor = None, stats: torch.Tensor = None) -> torch.Tensor:
        """Forward pass using only mel-spectrogram."""
        x = self.features(mel_spec)
        x = self.classifier(x)
        return x


def get_model(model_type: str = "efficient", num_classes: int = 3, dropout: float = 0.3) -> nn.Module:
    """
    Factory function to get the appropriate model.
    
    Args:
        model_type: "efficient" for dual-stream or "simple" for minimal RAM
        num_classes: Number of output classes
        dropout: Dropout rate
    
    Returns:
        Model instance
    """
    if model_type == "efficient":
        return EfficientAudioCNN(num_classes=num_classes, dropout=dropout)
    elif model_type == "simple":
        return SimplifiedCNN(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the models
    batch_size = 4
    n_mels = 128
    n_mfcc = 120
    time_steps = 259  # ~3 seconds at 44100Hz with hop_length=512
    n_stats = 11
    
    # Create dummy data
    mel_spec = torch.randn(batch_size, n_mels, time_steps)
    mfcc = torch.randn(batch_size, n_mfcc, time_steps)
    stats = torch.randn(batch_size, n_stats)
    
    # Test efficient model
    print("Testing EfficientAudioCNN:")
    efficient_model = get_model("efficient")
    output = efficient_model(mel_spec, mfcc, stats)
    print(f"  Input shapes: mel={mel_spec.shape}, mfcc={mfcc.shape}, stats={stats.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in efficient_model.parameters()):,}")
    
    # Test simplified model
    print("\nTesting SimplifiedCNN:")
    simple_model = get_model("simple")
    output = simple_model(mel_spec)
    print(f"  Input shape: mel={mel_spec.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in simple_model.parameters()):,}")

