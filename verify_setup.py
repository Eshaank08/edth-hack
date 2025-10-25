#!/usr/bin/env python3
"""
Verify that all components are properly installed and working.
Run this before training to catch any issues early.
"""

import sys
from pathlib import Path

print("🔍 Verifying Advanced Model Setup...\n")

# Check Python version
print("1. Checking Python version...")
if sys.version_info < (3, 10):
    print(f"   ❌ Python 3.10+ required, found {sys.version_info.major}.{sys.version_info.minor}")
    sys.exit(1)
else:
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Check required files
print("\n2. Checking required files...")
required_files = [
    "advanced_features.py",
    "advanced_model.py",
    "train_advanced_efficient.py",
    "competition_bot_advanced.py",
    "train_on_colab.ipynb",
]

all_files_present = True
for file in required_files:
    if Path(file).exists():
        print(f"   ✓ {file}")
    else:
        print(f"   ❌ {file} not found")
        all_files_present = False

if not all_files_present:
    print("\n❌ Some required files are missing!")
    sys.exit(1)

# Check dataset
print("\n3. Checking dataset...")
train_path = Path("data/raw/train")
val_path = Path("data/raw/val")

if train_path.exists():
    train_files = list(train_path.glob("**/*.wav"))
    print(f"   ✓ Training data found: {len(train_files)} audio files")
else:
    print(f"   ❌ Training data not found at {train_path}")
    print("      Download from: https://github.com/helsing-ai/edth-munich-drone-acoustics/releases")

if val_path.exists():
    val_files = list(val_path.glob("**/*.wav"))
    print(f"   ✓ Validation data found: {len(val_files)} audio files")
else:
    print(f"   ❌ Validation data not found at {val_path}")
    print("      Download from: https://github.com/helsing-ai/edth-munich-drone-acoustics/releases")

# Check dependencies
print("\n4. Checking dependencies...")
dependencies = {
    "torch": "PyTorch",
    "librosa": "Librosa",
    "soundfile": "SoundFile",
    "sklearn": "scikit-learn",
    "numpy": "NumPy",
    "tqdm": "tqdm",
}

missing_deps = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ❌ {name} not installed")
        missing_deps.append(name)

if missing_deps:
    print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
    print("   Install with: pip install torch librosa soundfile scikit-learn numpy tqdm")
    sys.exit(1)

# Test imports
print("\n5. Testing module imports...")
try:
    from advanced_features import AdvancedAudioFeatureExtractor
    print("   ✓ advanced_features")
except Exception as e:
    print(f"   ❌ advanced_features: {e}")
    sys.exit(1)

try:
    from advanced_model import get_model
    print("   ✓ advanced_model")
except Exception as e:
    print(f"   ❌ advanced_model: {e}")
    sys.exit(1)

# Test model creation
print("\n6. Testing model creation...")
try:
    import torch
    model = get_model("efficient", num_classes=3, dropout=0.3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ EfficientAudioCNN created ({n_params:,} parameters)")
    
    # Test forward pass
    batch_size = 2
    mel_spec = torch.randn(batch_size, 128, 259)
    mfcc = torch.randn(batch_size, 120, 259)
    stats = torch.randn(batch_size, 11)
    
    output = model(mel_spec, mfcc, stats)
    assert output.shape == (batch_size, 3), f"Expected shape ({batch_size}, 3), got {output.shape}"
    print(f"   ✓ Forward pass successful (output shape: {output.shape})")
except Exception as e:
    print(f"   ❌ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Check GPU availability
print("\n7. Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   ✓ GPU available: {gpu_name}")
        print(f"      Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   ⚠️  No GPU found - will use CPU (slower but works)")
except Exception as e:
    print(f"   ⚠️  Could not check GPU: {e}")

# Check models directory
print("\n8. Checking models directory...")
models_dir = Path("models")
if not models_dir.exists():
    models_dir.mkdir(parents=True)
    print(f"   ✓ Created {models_dir}")
else:
    print(f"   ✓ {models_dir} exists")

# Check for existing model
existing_model = models_dir / "best_model_advanced.pt"
if existing_model.exists():
    size_mb = existing_model.stat().st_size / (1024 * 1024)
    print(f"   ℹ️  Found existing model: {existing_model.name} ({size_mb:.2f} MB)")
    print("      You can use this with competition_bot_advanced.py")
else:
    print("   ℹ️  No trained model found yet")
    print("      Run train_advanced_efficient.py or use Google Colab to train")

# Summary
print("\n" + "="*60)
print("✅ Setup verification complete!")
print("="*60)
print("\n📋 Next Steps:")

if not (train_path.exists() and val_path.exists()):
    print("   1. Download dataset:")
    print("      curl -L -o data.zip https://github.com/helsing-ai/edth-munich-drone-acoustics/releases/download/train_val_data/drone_acoustics_train_val_data.zip")
    print("      unzip data.zip && mv train data/raw/ && mv val data/raw/")

if not existing_model.exists():
    print("\n   2. Train model (choose one):")
    print("      a) Google Colab (recommended):")
    print("         - Open train_on_colab.ipynb in Colab")
    print("         - Enable GPU and run all cells")
    print("         - Download trained model")
    print("\n      b) Local training:")
    print("         python train_advanced_efficient.py")

print("\n   3. Run competition bot:")
print("      python competition_bot_advanced.py")

print("\n💡 Tips:")
print("   - Use Google Colab if your laptop has limited RAM")
print("   - Training takes ~30 minutes on GPU, ~2 hours on CPU")
print("   - Expected validation accuracy: 85-92%")
print("   - Competition accuracy should be similar")

print("\n📚 Documentation:")
print("   - README_ADVANCED_SETUP.md - Complete setup guide")
print("   - QUICK_START_ADVANCED.md - Step-by-step instructions")
print("   - CLOUD_TRAINING_GUIDE.md - Cloud training options")

print("\n🚀 Good luck!\n")

