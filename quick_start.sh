#!/bin/bash
# Quick start script for the drone acoustics competition

set -e

echo "🚀 Drone Acoustics Competition - Quick Start"
echo "==========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "Please install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Check if virtual environment is set up
if [ ! -d ".venv" ]; then
    echo "📦 Setting up virtual environment..."
    uv sync
    echo "✅ Virtual environment created"
    echo ""
fi

# Check if dataset exists
if [ ! -d "data/raw/train" ] || [ ! -d "data/raw/val" ]; then
    echo "⚠️  Dataset not found!"
    echo ""
    echo "Please download the dataset:"
    echo "1. Download from: https://github.com/helsing-ai/edth-munich-drone-acoustics/releases/download/train_val_data/drone_acoustics_train_val_data.zip"
    echo ""
    echo "2. Extract and organize:"
    echo "   unzip drone_acoustics_train_val_data.zip"
    echo "   mkdir -p data/raw"
    echo "   mv train data/raw/"
    echo "   mv val data/raw/"
    echo ""
    read -p "Have you downloaded and extracted the dataset? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Please download the dataset first, then run this script again."
        exit 1
    fi
fi

# Check if model exists
if [ ! -f "models/best_model.pt" ]; then
    echo "🧠 Model not found. Training new model..."
    echo "This will take approximately 10-20 minutes."
    echo ""
    read -p "Start training now? (Y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        uv run python train_model.py
        echo ""
        echo "✅ Model training complete!"
    else
        echo "Please train the model first: uv run python train_model.py"
        exit 1
    fi
else
    echo "✅ Model found at models/best_model.pt"
fi

echo ""
echo "🎮 Starting competition bot..."
echo "Press Ctrl+C to stop"
echo ""
sleep 2

uv run python competition_bot.py

