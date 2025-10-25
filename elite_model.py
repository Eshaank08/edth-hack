#!/usr/bin/env python3
"""
Elite model architecture - maximum accuracy.
Multi-stream CNN with attention mechanism and deeper layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """Self-attention mechanism for feature refinement."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(x)
        return x * attention_weights


class EliteAudioCNN(nn.Module):
    """
    Elite multi-stream CNN for maximum accuracy.
    - Processes mel, MFCC, chroma, and spectral contrast separately
    - Uses attention mechanisms
    - Deeper architecture for better representation
    """
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        
        # Mel-spectrogram branch (128 x time)
        self.mel_branch = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.mel_attention = AttentionBlock(32)
        
        # MFCC branch (120 x time)
        self.mfcc_branch = nn.Sequential(
            nn.Conv1d(120, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(96, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.mfcc_attention = AttentionBlock(32)
        
        # Chroma branch (12 x time)
        self.chroma_branch = nn.Sequential(
            nn.Conv1d(12, 24, kernel_size=3, padding=1),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(24, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Spectral contrast branch (7 x time)
        self.contrast_branch = nn.Sequential(
            nn.Conv1d(7, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        # Fusion layer
        # 32 (mel) + 32 (mfcc) + 16 (chroma) + 16 (contrast) + 17 (stats) = 113
        fusion_dim = 32 + 32 + 16 + 16 + 17
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(32, num_classes),
        )
    
    def forward(
        self,
        mel_spec: torch.Tensor,
        mfcc: torch.Tensor,
        chroma: torch.Tensor,
        contrast: torch.Tensor,
        stats: torch.Tensor,
    ) -> torch.Tensor:
        # Process each stream
        mel_features = self.mel_branch(mel_spec)
        mel_features = mel_features.squeeze(-1)  # (batch, 32)
        
        mfcc_features = self.mfcc_branch(mfcc)
        mfcc_features = mfcc_features.squeeze(-1)  # (batch, 32)
        
        chroma_features = self.chroma_branch(chroma)
        chroma_features = chroma_features.squeeze(-1)  # (batch, 16)
        
        contrast_features = self.contrast_branch(contrast)
        contrast_features = contrast_features.squeeze(-1)  # (batch, 16)
        
        # Concatenate all features
        combined = torch.cat([
            mel_features,
            mfcc_features,
            chroma_features,
            contrast_features,
            stats
        ], dim=1)
        
        # Final classification
        output = self.fusion(combined)
        return output


def get_elite_model(num_classes: int = 3, dropout: float = 0.4) -> nn.Module:
    """Get elite model instance."""
    return EliteAudioCNN(num_classes=num_classes, dropout=dropout)


if __name__ == "__main__":
    # Test the model
    batch_size = 4
    time_steps = 259
    
    mel_spec = torch.randn(batch_size, 128, time_steps)
    mfcc = torch.randn(batch_size, 120, time_steps)
    chroma = torch.randn(batch_size, 12, time_steps)
    contrast = torch.randn(batch_size, 7, time_steps)
    stats = torch.randn(batch_size, 17)
    
    model = get_elite_model()
    output = model(mel_spec, mfcc, chroma, contrast, stats)
    
    print(f"Input shapes:")
    print(f"  mel_spec: {mel_spec.shape}")
    print(f"  mfcc: {mfcc.shape}")
    print(f"  chroma: {chroma.shape}")
    print(f"  contrast: {contrast.shape}")
    print(f"  stats: {stats.shape}")
    print(f"\nOutput shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

