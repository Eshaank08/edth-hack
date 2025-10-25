# 🚀 START HERE - Training Your Advanced Model

## ⚡ Quick Decision: Where to Train?

### 💻 Your Laptop Specs Unknown → Use Google Colab

**If your laptop has:**
- Less than 8 GB RAM → **Use Colab** ✅
- No dedicated GPU → **Use Colab** ✅  
- Want faster training → **Use Colab** ✅
- 16+ GB RAM + GPU → Can use local ⚡

---

## 🎯 Method 1: Google Colab (RECOMMENDED)

### Why Colab?
- ✅ **FREE GPU** - T4 GPU included
- ✅ **No RAM issues** - Runs in cloud
- ✅ **20-30 minutes** - Much faster than CPU
- ✅ **Easy setup** - Just upload and run

### Steps (5 minutes to start):

**1. Verify setup first:**
```bash
python verify_setup.py
```

**2. Open Google Colab:**
- Go to: https://colab.research.google.com/
- Click `File` → `Upload notebook`
- Select `train_on_colab.ipynb` from your laptop

**3. Enable GPU:**
- Click `Runtime` → `Change runtime type`
- Hardware accelerator: `T4 GPU`
- Click `Save`

**4. Run the notebook:**
- Click `Runtime` → `Run all`
- When prompted, upload these files from your laptop:
  - `advanced_features.py`
  - `advanced_model.py`
  - `train_advanced_efficient.py`

**5. Wait for training (~25 minutes):**
```
Training progress will show in output
Look for: ✓ Saved best model with val_acc: 0.XXXX
```

**6. Download model:**
- Last cell downloads `best_model_advanced.pt`
- Save it to your laptop's `models/` folder

**7. Run competition bot:**
```bash
python competition_bot_advanced.py
```

**DONE! 🎉**

---

## 💻 Method 2: Local Training

### Requirements:
- 4-8 GB free RAM
- 1-2 hours time (CPU) or 30 min (GPU)
- Python 3.10+

### Steps:

**1. Verify everything is ready:**
```bash
python verify_setup.py
```

**2. Check dataset exists:**
```bash
ls data/raw/train  # Should show: background/ drone/ helicopter/
ls data/raw/val    # Should show: background/ drone/ helicopter/
```

If missing:
```bash
curl -L -o data.zip https://github.com/helsing-ai/edth-munich-drone-acoustics/releases/download/train_val_data/drone_acoustics_train_val_data.zip
unzip data.zip
mkdir -p data/raw
mv train data/raw/
mv val data/raw/
```

**3. Start training:**
```bash
python train_advanced_efficient.py
```

You'll see:
```
🚀 Advanced Efficient Training Pipeline
========================================
Physical batch size: 8
Effective batch size: 32
Mixed precision: True
========================================
Loading datasets...
Training samples: 1234
Validation samples: 456
...
```

**4. Monitor progress:**
- Watch for validation accuracy increasing
- Best model auto-saves to `models/best_model_advanced.pt`
- Training stops early if no improvement for 10 epochs

**5. When complete, run bot:**
```bash
python competition_bot_advanced.py
```

### If you run out of RAM:

Edit `train_advanced_efficient.py` line 19:
```python
PHYSICAL_BATCH_SIZE = 4  # Change from 8 to 4 (or even 2)
```

Or just use Google Colab instead!

---

## 📊 What to Expect

### Training Output:
```
Epoch 1/50
==========================================================
Training: 100%|████████████| 156/156 [02:34<00:00]
Train Loss: 0.8234, Train Acc: 0.6523
Validating: 100%|██████████| 39/39 [00:21<00:00]
Val Loss: 0.6123, Val Acc: 0.7856
✓ Saved best model with val_acc: 0.7856

Epoch 2/50
...

Epoch 25/50
==========================================================
Train Loss: 0.2145, Train Acc: 0.9123
Val Loss: 0.3456, Val Acc: 0.8834
✓ Saved best model with val_acc: 0.8834

Training complete! Best validation accuracy: 0.8834
```

### Target Metrics:
- **Validation Accuracy:** 85-92% ✅
- **Training Time (Colab GPU):** 20-30 min ✅
- **Training Time (Laptop CPU):** 1-2 hours ⏰
- **Model Size:** ~400-500 KB ✅

---

## 🎮 After Training

### 1. Verify model exists:
```bash
ls -lh models/best_model_advanced.pt
# Should show ~400-500 KB file
```

### 2. Start competition bot:
```bash
python competition_bot_advanced.py
```

### 3. Watch it compete:
```
====================================================
New Challenge ID: xyz789
Downloading audio...
Download completed in 0.34s
Classifying audio...
Prediction: drone (confidence: 0.945)
Classification completed in 0.52s

====================================================
Result: Correct!
✓ Score Awarded: 186
✓ Total Score: 1847
Stats: 47/51 correct (92.2%)
====================================================
```

---

## 🆘 Troubleshooting

### ❌ "Out of memory" on laptop
**Solution:** Use Google Colab (free GPU, no local RAM usage)

### ❌ "Dataset not found"
```bash
# Download dataset
curl -L -o data.zip https://github.com/helsing-ai/edth-munich-drone-acoustics/releases/download/train_val_data/drone_acoustics_train_val_data.zip
unzip data.zip
mv train data/raw/ && mv val data/raw/
```

### ❌ "Module not found"
```bash
# Install dependencies
pip install torch librosa soundfile scikit-learn numpy tqdm
```

### ❌ Colab disconnected during training
- Download model periodically (it saves every time val accuracy improves)
- Use browser extension "Keep Colab Alive"
- Or just restart from where it stopped

### ❌ Low accuracy after training
- Make sure you have enough training data (100+ files per class)
- Try training longer (increase epochs)
- Or retrain - sometimes random seed matters

---

## 💡 Pro Tips

1. **First time?** Use Colab - it's easier and faster
2. **Keep terminal open** during local training to monitor
3. **Download model immediately** after Colab training
4. **Test bot first** before leaving it running
5. **Monitor logs** to learn from mistakes

---

## 📈 Expected Competition Performance

With this advanced model:
- **Accuracy:** 85-92% (vs 75% baseline)
- **Speed:** ~1 second per challenge
- **Score per correct:** ~180 points (out of 200 max)
- **Improvement:** +40 points per challenge vs baseline

---

## ⏱️ Time Commitment

| Task | Colab | Local (CPU) | Local (GPU) |
|------|-------|-------------|-------------|
| Setup | 5 min | 2 min | 2 min |
| Training | 25 min | 120 min | 30 min |
| **Total** | **30 min** | **2 hours** | **32 min** |

---

## 🎯 One-Line Commands

### Verify setup:
```bash
python verify_setup.py
```

### Train locally:
```bash
python train_advanced_efficient.py
```

### Run bot:
```bash
python competition_bot_advanced.py
```

### Monitor (in separate terminal):
```bash
watch -n 5 'tail -20 nohup.out'  # If running bot in background
```

---

## 📚 Need More Help?

- **README_ADVANCED_SETUP.md** - Complete technical guide
- **QUICK_START_ADVANCED.md** - Detailed walkthrough
- **CLOUD_TRAINING_GUIDE.md** - All cloud platform options
- **COMPETITION_GUIDE.md** - API and rules

---

## 🏆 Success Checklist

- [ ] Ran `python verify_setup.py` - all checks passed
- [ ] Downloaded dataset to `data/raw/train` and `data/raw/val`
- [ ] Chose training method (Colab recommended)
- [ ] Trained model successfully
- [ ] Saved model to `models/best_model_advanced.pt`
- [ ] Ran `python competition_bot_advanced.py`
- [ ] Seeing predictions and scores
- [ ] Validation accuracy > 85%
- [ ] Competition accuracy > 80%

---

## 🚀 Ready? Let's Go!

```bash
# Step 1: Verify
python verify_setup.py

# Step 2: Train (choose one)
# Option A: Use Colab (upload train_on_colab.ipynb)
# Option B: python train_advanced_efficient.py

# Step 3: Compete!
python competition_bot_advanced.py
```

**Good luck! 🎯🏆**

