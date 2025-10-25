#!/usr/bin/env python3
"""
Automated competition bot for real-time drone acoustics challenge.
Polls for new challenges, downloads audio, runs inference, and submits predictions.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from hs_hackathon_drone_acoustics import CLASSES
from hs_hackathon_drone_acoustics.base import AudioWaveform
from hs_hackathon_drone_acoustics.feature_extractors import MFCCFeatureExtractor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "https://edth.helsing.codes"
AUTH_TOKEN = "f276bbf9-e42b-452c-be54-eac3d4c6f0e3"
MODEL_PATH = Path(__file__).parent / "models" / "best_model.pt"
TEMP_AUDIO_PATH = Path(__file__).parent / "temp_challenge.wav"


class AudioClassifier(nn.Module):
    """Neural network classifier for audio classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CompetitionBot:
    """Automated bot for the drone acoustics competition."""

    def __init__(self, auth_token: str, model_path: Path):
        self.auth_token = auth_token
        self.model_path = model_path
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json",
            }
        )

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.feature_extractor = self._load_model()
        logger.info(f"Model loaded successfully on {self.device}")

        self.last_challenge_id: str | None = None
        self.total_score = 0
        self.challenges_attempted = 0
        self.challenges_correct = 0

    def _load_model(self) -> tuple[AudioClassifier, Any]:
        """Load the trained model and feature extractor."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. Please train the model first using train_model.py"
            )

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        # Initialize model
        model = AudioClassifier(
            input_dim=checkpoint["input_dim"], hidden_dim=128, num_classes=len(CLASSES)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        # Initialize feature extractor
        mfcc_extractor = MFCCFeatureExtractor(n_mfcc=checkpoint["n_mfcc"])
        scaler = StandardScaler()
        scaler.mean_ = checkpoint["scaler_mean"]
        scaler.scale_ = checkpoint["scaler_scale"]

        feature_extractor = {"mfcc": mfcc_extractor, "scaler": scaler}

        logger.info(f"Model validation accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
        return model, feature_extractor

    def get_current_challenge(self) -> dict[str, Any] | None:
        """Fetch the current challenge from the API."""
        try:
            response = self.session.get(f"{API_BASE_URL}/api/challenge", timeout=5)
            response.raise_for_status()
            challenge = response.json()
            return challenge
        except requests.RequestException as e:
            logger.error(f"Error fetching challenge: {e}")
            return None

    def download_audio(self, wav_url: str, save_path: Path) -> bool:
        """Download audio file from the given URL."""
        try:
            full_url = f"{API_BASE_URL}{wav_url}"
            response = self.session.get(full_url, timeout=10)
            response.raise_for_status()

            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
        except requests.RequestException as e:
            logger.error(f"Error downloading audio: {e}")
            return False

    def classify_audio(self, audio_path: Path) -> str:
        """Classify the audio file and return the predicted class."""
        try:
            # Load waveform
            waveform = AudioWaveform.load(audio_path)

            # Extract features
            mfcc_features = self.feature_extractor["mfcc"].extract(waveform)
            scaled_features = self.feature_extractor["scaler"].transform(
                mfcc_features.reshape(1, -1)
            )
            features_tensor = torch.from_numpy(scaled_features).float().to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = outputs.argmax(1).item()

            prediction = CLASSES[predicted_class]
            confidence = probabilities[0][predicted_class].item()

            logger.info(
                f"Prediction: {prediction} (confidence: {confidence:.3f})"
            )
            logger.info(
                f"All probabilities: {', '.join([f'{CLASSES[i]}: {probabilities[0][i]:.3f}' for i in range(len(CLASSES))])}"
            )

            return prediction
        except Exception as e:
            logger.error(f"Error classifying audio: {e}")
            # Default to most common class if error
            return "background"

    def submit_classification(
        self, challenge_id: str, classification: str
    ) -> dict[str, Any] | None:
        """Submit classification to the API."""
        try:
            payload = {
                "challenge_id": challenge_id,
                "classification": classification,
            }
            response = self.session.post(
                f"{API_BASE_URL}/api/challenge", json=payload, timeout=5
            )
            response.raise_for_status()
            result = response.json()
            return result
        except requests.RequestException as e:
            logger.error(f"Error submitting classification: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            return None

    def process_challenge(self, challenge: dict[str, Any]) -> None:
        """Process a single challenge: download, classify, and submit."""
        challenge_id = challenge["challenge_id"]
        wav_url = challenge["wav_url"]
        time_remaining = challenge.get("time_until_next_rotation_ms", 0) / 1000

        logger.info(f"\n{'='*60}")
        logger.info(f"New Challenge ID: {challenge_id}")
        logger.info(f"Time until rotation: {time_remaining:.1f}s")
        logger.info(f"{'='*60}")

        start_time = time.time()

        # Download audio
        logger.info("Downloading audio...")
        if not self.download_audio(wav_url, TEMP_AUDIO_PATH):
            logger.error("Failed to download audio. Skipping challenge.")
            return

        download_time = time.time() - start_time
        logger.info(f"Download completed in {download_time:.2f}s")

        # Classify audio
        logger.info("Classifying audio...")
        classification_start = time.time()
        prediction = self.classify_audio(TEMP_AUDIO_PATH)
        classification_time = time.time() - classification_start
        logger.info(f"Classification completed in {classification_time:.2f}s")

        # Submit classification
        logger.info(f"Submitting prediction: {prediction}")
        result = self.submit_classification(challenge_id, prediction)

        total_time = time.time() - start_time

        if result:
            self.challenges_attempted += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Result: {result.get('message', 'Unknown')}")
            if result.get("success"):
                self.challenges_correct += 1
                score_awarded = result.get("score_awarded", 0)
                self.total_score = result.get("total_score", self.total_score)
                logger.info(f"✓ Score Awarded: {score_awarded}")
                logger.info(f"✓ Total Score: {self.total_score}")
            else:
                logger.info(f"✗ Incorrect prediction")
            logger.info(f"Total time: {total_time:.2f}s")
            logger.info(
                f"Stats: {self.challenges_correct}/{self.challenges_attempted} correct "
                f"({100*self.challenges_correct/self.challenges_attempted:.1f}%)"
            )
            logger.info(f"{'='*60}\n")
        else:
            logger.error("Failed to submit classification")

        # Cleanup
        if TEMP_AUDIO_PATH.exists():
            TEMP_AUDIO_PATH.unlink()

    def run(self, poll_interval: float = 2.0) -> None:
        """Main loop: continuously poll for new challenges and process them."""
        logger.info("Starting competition bot...")
        logger.info(f"Polling interval: {poll_interval}s")
        logger.info("Press Ctrl+C to stop\n")

        try:
            while True:
                # Get current challenge
                challenge = self.get_current_challenge()

                if challenge:
                    challenge_id = challenge["challenge_id"]

                    # Check if it's a new challenge
                    if challenge_id != self.last_challenge_id:
                        self.last_challenge_id = challenge_id
                        self.process_challenge(challenge)
                    else:
                        time_remaining = (
                            challenge.get("time_until_next_rotation_ms", 0) / 1000
                        )
                        logger.info(
                            f"Waiting for new challenge... (next rotation in {time_remaining:.1f}s)"
                        )

                # Wait before next poll
                time.sleep(poll_interval)

        except KeyboardInterrupt:
            logger.info("\n\nBot stopped by user")
            logger.info(f"Final Stats:")
            logger.info(
                f"  Challenges attempted: {self.challenges_attempted}"
            )
            logger.info(
                f"  Challenges correct: {self.challenges_correct}"
            )
            if self.challenges_attempted > 0:
                accuracy = 100 * self.challenges_correct / self.challenges_attempted
                logger.info(f"  Accuracy: {accuracy:.1f}%")
            logger.info(f"  Total score: {self.total_score}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise


def main() -> None:
    """Main entry point."""
    bot = CompetitionBot(auth_token=AUTH_TOKEN, model_path=MODEL_PATH)
    bot.run(poll_interval=2.0)


if __name__ == "__main__":
    main()

