#!/usr/bin/env python3
"""
Complete setup and run script for the competition.
Handles dataset download, model training, and bot launching.
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

import requests


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def check_uv() -> bool:
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def setup_venv() -> None:
    """Set up virtual environment."""
    if not Path(".venv").exists():
        print("📦 Setting up virtual environment...")
        subprocess.run(["uv", "sync"], check=True)
        print("✅ Virtual environment created\n")
    else:
        print("✅ Virtual environment already exists\n")


def download_dataset() -> bool:
    """Download and extract the dataset."""
    train_path = Path("data/raw/train")
    val_path = Path("data/raw/val")

    if train_path.exists() and val_path.exists():
        print("✅ Dataset already downloaded\n")
        return True

    print("📥 Downloading dataset (this may take a few minutes)...")
    dataset_url = "https://github.com/helsing-ai/edth-munich-drone-acoustics/releases/download/train_val_data/drone_acoustics_train_val_data.zip"
    zip_path = Path("drone_acoustics_train_val_data.zip")

    try:
        # Download with progress
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(zip_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end="", flush=True)

        print("\n✅ Download complete")

        # Extract
        print("📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")

        # Organize
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        if Path("train").exists():
            Path("train").rename("data/raw/train")
        if Path("val").exists():
            Path("val").rename("data/raw/val")

        # Cleanup
        zip_path.unlink()

        print("✅ Dataset extracted and organized\n")
        return True

    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nPlease download manually from:")
        print(dataset_url)
        print("\nThen extract and place train/ and val/ in data/raw/")
        return False


def train_model(quick: bool = False) -> bool:
    """Train the model."""
    model_path = Path("models/best_model.pt")

    if model_path.exists():
        print("✅ Model already exists\n")
        response = input("Do you want to retrain? (y/N): ").strip().lower()
        if response != "y":
            return True

    if quick:
        print("🚀 Creating quick baseline model for testing...")
        print("⚠️  This will have low accuracy. Use full training for competition!\n")
        script = "train_baseline_quick.py"
    else:
        print("🧠 Training model (this will take 10-20 minutes)...")
        script = "train_model.py"

    try:
        subprocess.run(["uv", "run", "python", script], check=True)
        print("\n✅ Model training complete!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error training model: {e}")
        return False


def run_bot() -> None:
    """Run the competition bot."""
    print_header("🎮 Starting Competition Bot")
    print("Press Ctrl+C to stop\n")

    try:
        subprocess.run(["uv", "run", "python", "competition_bot.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n✅ Bot stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running bot: {e}")


def main() -> None:
    """Main setup and run process."""
    print_header("🚀 Drone Acoustics Competition Setup")

    # Check uv
    if not check_uv():
        print("❌ Error: uv is not installed")
        print("Please install: https://docs.astral.sh/uv/getting-started/installation/")
        sys.exit(1)

    print("✅ uv is installed\n")

    # Setup virtual environment
    setup_venv()

    # Check for dataset
    train_exists = Path("data/raw/train").exists()
    val_exists = Path("data/raw/val").exists()

    if not train_exists or not val_exists:
        print("⚠️  Dataset not found")
        response = input("Download dataset now? (Y/n): ").strip().lower()
        if response != "n":
            if not download_dataset():
                # If download fails, offer quick baseline
                print("\n⚠️  Couldn't download dataset")
                response = input("Create quick baseline model for testing? (Y/n): ").strip().lower()
                if response != "n":
                    if not train_model(quick=True):
                        sys.exit(1)
                else:
                    sys.exit(1)
        else:
            # User chose not to download, offer quick baseline
            response = input("Create quick baseline model for testing? (Y/n): ").strip().lower()
            if response != "n":
                if not train_model(quick=True):
                    sys.exit(1)
            else:
                sys.exit(1)
    else:
        print("✅ Dataset found\n")

    # Train model (only if we have the full dataset and no quick model was made)
    if (train_exists and val_exists) and not Path("models/best_model.pt").exists():
        response = input("Train model now? (Y/n): ").strip().lower()
        if response != "n":
            if not train_model(quick=False):
                sys.exit(1)
        else:
            # Offer quick baseline
            response = input("Create quick baseline model instead? (Y/n): ").strip().lower()
            if response != "n":
                if not train_model(quick=True):
                    sys.exit(1)
            else:
                print("Please train a model before running the bot")
                sys.exit(1)

    # Check if model exists
    if not Path("models/best_model.pt").exists():
        print("❌ No model found. Please train a model first.")
        sys.exit(1)

    # Run the bot
    response = input("Start the competition bot now? (Y/n): ").strip().lower()
    if response != "n":
        run_bot()
    else:
        print("\n✅ Setup complete!")
        print("Run the bot anytime with: uv run python competition_bot.py")


if __name__ == "__main__":
    main()

