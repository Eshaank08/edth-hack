# 🚀 Competition Quick Start Guide

Your credentials:
- **Username**: eshaank08
- **Email**: eshaankansal0@gmail.com
- **Token**: f276bbf9-e42b-452c-be54-eac3d4c6f0e3

## 🎯 Quick Start (3 Steps)

### Step 1: Download the Dataset

Download the training data:
```bash
curl -L -o drone_acoustics_train_val_data.zip https://github.com/helsing-ai/edth-munich-drone-acoustics/releases/download/train_val_data/drone_acoustics_train_val_data.zip
```

Extract and organize:
```bash
unzip drone_acoustics_train_val_data.zip
mkdir -p data/raw
mv train data/raw/
mv val data/raw/
```

### Step 2: Train Your Model

Train the classifier (takes ~10-20 minutes):
```bash
uv run python train_model.py
```

This will:
- Load the training and validation datasets
- Extract MFCC features from audio
- Train a neural network classifier
- Save the best model to `models/best_model.pt`

### Step 3: Start the Competition Bot

Launch the automated bot:
```bash
uv run python competition_bot.py
```

The bot will:
- ✅ Poll for new challenges every 2 seconds
- ✅ Download audio files instantly
- ✅ Run inference using your trained model
- ✅ Submit predictions automatically
- ✅ Show live scores and accuracy

**That's it!** The bot will run continuously and compete for you.

---

## 📊 What Each Script Does

### `train_model.py`
- Trains a neural network on MFCC features
- Uses PyTorch with dropout for regularization
- Saves best model based on validation accuracy
- Shows confusion matrix and metrics

### `competition_bot.py`
- Automated competition client
- Polls API every 2 seconds for new challenges
- Downloads audio, runs inference, submits prediction
- Tracks score, accuracy, and stats in real-time

---

## 🎮 Manual Testing (Optional)

Test the API manually:

```bash
# Check current challenge
curl https://edth.helsing.codes/api/challenge

# Download audio
curl https://edth.helsing.codes/wavs/CHALLENGE_ID.wav -o test.wav

# Submit prediction
curl -X POST https://edth.helsing.codes/api/challenge \
  -H "Authorization: Bearer f276bbf9-e42b-452c-be54-eac3d4c6f0e3" \
  -H "Content-Type: application/json" \
  -d '{"challenge_id": "YOUR_CHALLENGE_ID", "classification": "drone"}'
```

---

## 📈 Live Viewer

Watch the competition live:
- **URL**: https://edth.helsing.codes/static/index.html
- See real-time spectrograms
- View leaderboard
- Monitor other participants

---

## 🏆 Scoring System

- **Base**: 100 points for correct answer
- **Speed Bonus**: Up to 100 extra points (faster = more points)
- **Penalty**: 0 points for wrong answer
- **Limit**: One submission per challenge

---

## 🎯 Strategy Tips

1. **Speed is Key**: The bot processes in ~1-2 seconds to maximize speed bonus
2. **Model Quality**: Better accuracy = more points over time
3. **Uptime**: Keep the bot running to catch all challenges
4. **Monitor**: Watch the logs to track performance

---

## 🔧 Troubleshooting

### Model not found error
```bash
# Make sure you trained the model first
uv run python train_model.py
```

### Dataset not found error
```bash
# Download and extract the dataset
ls data/raw/train  # Should show background/, drone/, helicopter/
ls data/raw/val    # Should show background/, drone/, helicopter/
```

### API errors
- Check your internet connection
- Verify the token is correct
- Wait a few seconds and try again

---

## 📝 Next Steps

Want to improve your score?

1. **Better Features**: Try spectrograms, chromagrams, or mel-spectrograms
2. **Bigger Model**: Increase hidden layer sizes or add more layers
3. **Ensemble**: Train multiple models and vote on predictions
4. **Data Augmentation**: Add noise, pitch shifts, time stretches during training
5. **Fine-tuning**: Adjust learning rate, batch size, or regularization

Good luck! 🚀

