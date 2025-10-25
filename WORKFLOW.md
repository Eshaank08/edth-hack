# 🎯 Competition Bot Workflow

## 📊 Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPETITION SYSTEM                        │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  STEP 1: SETUP & PREPARATION                                 │
└──────────────────────────────────────────────────────────────┘

    Option A: Quick Test              Option B: Full Competition
    ================                  ======================
    
    train_baseline_quick.py           1. Download dataset
           ↓                                   ↓
    Creates model in 10s              2. train_model.py
           ↓                                   ↓
    models/best_model.pt              Trains for 20 min
    (Low accuracy ~33%)                      ↓
                                      models/best_model.pt
                                      (High accuracy 70-85%)

┌──────────────────────────────────────────────────────────────┐
│  STEP 2: RUN COMPETITION BOT                                 │
└──────────────────────────────────────────────────────────────┘

    uv run python competition_bot.py
           ↓
    ┌──────────────────┐
    │  Bot Starts      │
    │  - Load model    │
    │  - Initialize    │
    └────────┬─────────┘
             ↓
    ┌──────────────────┐
    │  Main Loop       │ ←─────────────┐
    │  (Every 2s)      │               │
    └────────┬─────────┘               │
             ↓                          │
    ┌──────────────────┐               │
    │  Poll API        │               │
    │  GET /challenge  │               │
    └────────┬─────────┘               │
             ↓                          │
       New Challenge?                   │
       /           \                    │
     YES           NO                   │
      ↓             └──────────────────┤
    ┌──────────────────┐               │
    │  Download Audio  │               │
    │  (0.5-1s)        │               │
    └────────┬─────────┘               │
             ↓                          │
    ┌──────────────────┐               │
    │  Extract MFCC    │               │
    │  Features        │               │
    └────────┬─────────┘               │
             ↓                          │
    ┌──────────────────┐               │
    │  Run Inference   │               │
    │  (< 1s)          │               │
    └────────┬─────────┘               │
             ↓                          │
    ┌──────────────────┐               │
    │  Submit          │               │
    │  Prediction      │               │
    │  POST /challenge │               │
    └────────┬─────────┘               │
             ↓                          │
    ┌──────────────────┐               │
    │  Get Result      │               │
    │  - Score         │               │
    │  - Correct?      │               │
    └────────┬─────────┘               │
             ↓                          │
    ┌──────────────────┐               │
    │  Update Stats    │               │
    │  Log Result      │               │
    └────────┬─────────┘               │
             └───────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  TIMING BREAKDOWN                                            │
└──────────────────────────────────────────────────────────────┘

    Challenge appears (t=0s)
           ↓ 0.1s
    Bot detects new challenge
           ↓ 0.5-1s
    Download audio
           ↓ 0.2s
    Extract features
           ↓ 0.1s
    Run inference
           ↓ 0.1s
    Submit prediction
           ↓
    Total: ~1-2 seconds
    
    Result: Maximum speed bonus! (~95+ bonus points)

┌──────────────────────────────────────────────────────────────┐
│  SYSTEM ARCHITECTURE                                         │
└──────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────┐
    │     competition_bot.py              │
    │  ┌────────────────────────────────┐ │
    │  │  CompetitionBot Class          │ │
    │  │  - API client                  │ │
    │  │  - Model inference             │ │
    │  │  - Score tracking              │ │
    │  └────────────────────────────────┘ │
    └──────────┬──────────────────────────┘
               │
               ├─→ Helsing API
               │   (https://edth.helsing.codes)
               │
               ├─→ AudioClassifier (PyTorch)
               │   - 3-layer neural network
               │   - Input: 40 MFCC features
               │   - Output: 3 classes
               │
               └─→ Feature Extractors
                   - MFCC extraction
                   - StandardScaler

┌──────────────────────────────────────────────────────────────┐
│  DATA FLOW                                                   │
└──────────────────────────────────────────────────────────────┘

    Audio File (.wav)
           ↓
    [AudioWaveform.load()]
           ↓
    Waveform Data + Sample Rate
           ↓
    [MFCCFeatureExtractor.extract()]
           ↓
    MFCC Features (40 coefficients)
           ↓
    [StandardScaler.transform()]
           ↓
    Normalized Features
           ↓
    [AudioClassifier.forward()]
           ↓
    Logits [background, drone, helicopter]
           ↓
    [Softmax + Argmax]
           ↓
    Prediction: "drone" (with confidence)

┌──────────────────────────────────────────────────────────────┐
│  SCORING SYSTEM                                              │
└──────────────────────────────────────────────────────────────┘

    Submission @ t seconds
           ↓
    Base Score: 100 (if correct)
           ↓
    Speed Bonus: max(0, 100 - t)
           ↓
    Total Score = 100 + bonus
    
    Examples:
    - t=1s:  100 + 99 = 199 points ⭐⭐⭐
    - t=5s:  100 + 95 = 195 points ⭐⭐
    - t=50s: 100 + 50 = 150 points ⭐
    - Wrong:            0 points   ❌

┌──────────────────────────────────────────────────────────────┐
│  FILE SYSTEM                                                 │
└──────────────────────────────────────────────────────────────┘

    edth-munich-drone-acoustics/
    │
    ├── competition_bot.py       ← Run this!
    ├── train_model.py
    ├── train_baseline_quick.py
    │
    ├── models/
    │   └── best_model.pt        ← Loaded by bot
    │
    ├── data/
    │   ├── examples/            ← Sample files
    │   └── raw/
    │       ├── train/           ← Training data
    │       └── val/             ← Validation data
    │
    └── src/
        └── hs_hackathon_drone_acoustics/
            ├── base.py          ← AudioWaveform
            ├── feature_extractors.py
            └── metrics.py

┌──────────────────────────────────────────────────────────────┐
│  MODEL ARCHITECTURE                                          │
└──────────────────────────────────────────────────────────────┘

    Input: [batch_size, 40]
           ↓
    Linear(40 → 128) + ReLU + Dropout(0.3)
           ↓
    Linear(128 → 128) + ReLU + Dropout(0.3)
           ↓
    Linear(128 → 64) + ReLU + Dropout(0.2)
           ↓
    Linear(64 → 3)
           ↓
    Output: [batch_size, 3] (logits)

    Softmax → Probabilities
    Argmax → Predicted class (0=background, 1=drone, 2=helicopter)

┌──────────────────────────────────────────────────────────────┐
│  ERROR HANDLING                                              │
└──────────────────────────────────────────────────────────────┘

    API Connection Error
           ↓
    Log error, wait 2s, retry
    
    Audio Download Error
           ↓
    Log error, skip challenge
    
    Model Inference Error
           ↓
    Use fallback prediction ("background")
    
    Submission Error
           ↓
    Log error with response details

┌──────────────────────────────────────────────────────────────┐
│  MONITORING & LOGGING                                        │
└──────────────────────────────────────────────────────────────┘

    Bot logs show:
    
    [INFO] New Challenge ID: xxx
    [INFO] Downloading audio...
    [INFO] Download completed in 0.8s
    [INFO] Classifying audio...
    [INFO] Prediction: drone (confidence: 0.85)
    [INFO] All probabilities: background: 0.10, drone: 0.85, ...
    [INFO] Submitting prediction: drone
    [INFO] Result: Correct!
    [INFO] ✓ Score Awarded: 195
    [INFO] ✓ Total Score: 450
    [INFO] Stats: 3/3 correct (100.0%)

┌──────────────────────────────────────────────────────────────┐
│  QUICK START COMMANDS                                        │
└──────────────────────────────────────────────────────────────┘

    # Test everything
    uv run python test_api.py
    
    # Quick test (2 min)
    uv run python train_baseline_quick.py
    uv run python competition_bot.py
    
    # Full competition (30 min setup)
    uv run python train_model.py
    uv run python competition_bot.py
    
    # Easy mode
    uv run python setup_and_run.py

┌──────────────────────────────────────────────────────────────┐
│  SUCCESS METRICS                                             │
└──────────────────────────────────────────────────────────────┘

    Target Performance:
    
    ✓ Submission time: < 2 seconds
    ✓ Model accuracy: 70-85%
    ✓ Speed bonus: 95+ points
    ✓ Uptime: 100% (bot runs continuously)
    ✓ Error rate: < 1%
    
    Expected Points Per Challenge:
    - Baseline model: ~130 (33% acc × 100 + 95 speed)
    - Trained model:  ~175 (75% acc × 100 + 95 speed)

┌──────────────────────────────────────────────────────────────┐
│  COMPETITION TIMELINE                                        │
└──────────────────────────────────────────────────────────────┘

    Challenge 1: t=0s    → t=100s  (rotate)
    Challenge 2: t=100s  → t=200s  (rotate)
    Challenge 3: t=200s  → t=300s  (rotate)
    ...
    
    Bot participates in ALL challenges
    No manual intervention needed
    Runs 24/7 if desired

┌──────────────────────────────────────────────────────────────┐
│  OPTIMIZATION OPPORTUNITIES                                  │
└──────────────────────────────────────────────────────────────┘

    Current:        Future Improvements:
    
    MFCC (40)    →  MFCC (60-80)
    2 layers     →  3-4 layers
    128 hidden   →  256 hidden
    No ensemble  →  Ensemble models
    Basic aug.   →  Advanced augmentation
    
    Expected accuracy gain: 70% → 85%+
```

## 🎯 Key Points

1. **Bot is fully automated** - No manual intervention needed
2. **Speed optimized** - Submits in 1-2 seconds for max bonus
3. **Error resilient** - Handles API errors gracefully
4. **Continuously running** - Catches every challenge
5. **Real-time logging** - See what's happening
6. **Score tracking** - Monitor your performance

## 🚀 Ready to Compete!

Your system is production-ready and optimized for the competition. Just run it and watch the points accumulate! 🏆

