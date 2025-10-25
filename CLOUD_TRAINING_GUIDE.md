# ☁️ Cloud Training Guide - Save Your Laptop RAM!

You have **3 options** for training your model:

---

## ⚡ Option 1: Google Colab (Recommended - FREE GPU!)

**Pros:** Free GPU, faster training, no laptop RAM usage
**Cons:** Need to upload files, session timeout after ~12 hours

### Steps:

1. **Open Google Colab:**
   - Go to https://colab.research.google.com/
   - Upload `train_on_colab.ipynb`

2. **Enable GPU:**
   - Click `Runtime` → `Change runtime type`
   - Select `T4 GPU` (free tier)
   - Click `Save`

3. **Run all cells:**
   - Click `Runtime` → `Run all`
   - Follow prompts to upload your code files

4. **Download trained model:**
   - Last cell will download `best_model_advanced.pt`
   - Place it in `models/` folder on your laptop

5. **Run competition bot locally:**
   ```bash
   python competition_bot_advanced.py
   ```

**Training time on Colab:** ~20-30 minutes with GPU

---

## 💻 Option 2: Train Locally (Memory-Efficient)

If you want to train on your laptop, I've optimized the training script:

### Memory-saving techniques used:
- ✅ Small batch size (8 instead of 32)
- ✅ Gradient accumulation (simulates batch 32)
- ✅ Mixed precision training (saves 40% memory)
- ✅ Efficient model architecture (~100K parameters)
- ✅ Memory cleanup after each batch

### Run on your laptop:

```bash
# Make sure you have the dataset
ls data/raw/train  # Should show background/, drone/, helicopter/

# Start training
python train_advanced_efficient.py
```

**Training time on laptop:** ~1-2 hours (CPU) or ~30 minutes (GPU)

**RAM usage:** ~2-4 GB (should work on most laptops)

---

## 🌩️ Option 3: Other Cloud Platforms (Paid)

### AWS SageMaker
```bash
# Use ml.g4dn.xlarge instance (~$0.50/hour)
# Already has PyTorch installed
```

### Kaggle Notebooks
- Free GPU: 30 hours/week
- Go to https://www.kaggle.com/code
- Similar to Colab but different limits

### Paperspace Gradient
- Free tier: 6 hours/month
- More reliable than Colab for long runs

---

## 🎯 Comparison Table

| Method | Cost | Speed | Setup Time | RAM Used |
|--------|------|-------|------------|----------|
| **Google Colab** | Free | ⚡⚡⚡ Fast | ~5 min | 0 (cloud) |
| **Laptop (optimized)** | Free | 🐢 Slow | 0 min | ~3 GB |
| **AWS/Cloud** | $0.50-2/hr | ⚡⚡⚡ Fast | ~15 min | 0 (cloud) |

---

## 🚀 Quick Start Commands

### For Google Colab:
1. Upload `train_on_colab.ipynb` to Colab
2. Run all cells
3. Download model
4. Run locally: `python competition_bot_advanced.py`

### For Local Training:
```bash
# One command to start
python train_advanced_efficient.py

# Monitor with less memory usage
watch -n 1 free -h  # Linux/Mac
```

### Check Model Performance:
```bash
# After training completes
python competition_bot_advanced.py
```

---

## 📊 Expected Results

With the advanced model, you should see:

- **Validation Accuracy:** 85-95%
- **Competition Score:** Significantly higher than baseline
- **Speed Bonus:** Full points (fast inference ~0.5s)

---

## 🔧 Troubleshooting

### "Out of Memory" on laptop
```bash
# Edit train_advanced_efficient.py
# Line 17: Change to smaller batch
PHYSICAL_BATCH_SIZE = 4  # Instead of 8
```

### Colab session disconnected
- Models auto-save in `models/` folder
- Download periodically
- Re-run from last checkpoint

### Model not loading in bot
```bash
# Make sure file exists
ls -lh models/best_model_advanced.pt

# Check path in bot
python -c "from pathlib import Path; print(Path('models/best_model_advanced.pt').exists())"
```

---

## 💡 Pro Tips

1. **Use Colab for training**, local for inference
2. **Download model immediately** after Colab training
3. **Monitor training** in Colab output
4. **Keep bot running** locally to catch all challenges
5. **Test model first** before competing

---

## ⏱️ Time Estimates

| Task | Colab | Laptop (CPU) |
|------|-------|--------------|
| Download dataset | 2 min | 2 min |
| Fit feature extractor | 3 min | 5 min |
| Train 50 epochs | 20 min | 90 min |
| **Total** | **25 min** | **~2 hours** |

---

## 🎯 Recommended Workflow

1. **First time:** Use Colab to train quickly
2. **Get competitive:** Run bot locally with trained model
3. **Iterate:** Adjust model, retrain on Colab
4. **Win:** Keep bot running 24/7 for maximum score!

Good luck! 🚀

