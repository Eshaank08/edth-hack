.PHONY: help test quick train compete setup install clean

help:
	@echo "🚀 Drone Acoustics Competition - Available Commands"
	@echo ""
	@echo "Quick Start:"
	@echo "  make quick      - Create baseline model and start competing (2 min)"
	@echo "  make compete    - Start the competition bot"
	@echo ""
	@echo "Full Setup:"
	@echo "  make install    - Install dependencies with uv"
	@echo "  make train      - Train the full model (requires dataset)"
	@echo "  make setup      - Interactive setup with dataset download"
	@echo ""
	@echo "Testing:"
	@echo "  make test       - Test API connectivity"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean      - Clean temporary files"
	@echo "  make status     - Show current status"
	@echo ""
	@echo "Your credentials:"
	@echo "  Username: eshaank08"
	@echo "  Token: f276bbf9-e42b-452c-be54-eac3d4c6f0e3"

install:
	@echo "📦 Installing dependencies..."
	uv sync
	@echo "✅ Dependencies installed"

test:
	@echo "🧪 Testing API connectivity..."
	uv run python test_api.py

quick:
	@echo "🚀 Quick start: Creating baseline model and competing..."
	uv run python train_baseline_quick.py
	@echo ""
	@echo "Starting competition bot..."
	uv run python competition_bot.py

train:
	@echo "🧠 Training full model..."
	@if [ ! -d "data/raw/train" ]; then \
		echo "❌ Dataset not found. Please download first:"; \
		echo "   make setup"; \
		exit 1; \
	fi
	uv run python train_model.py

compete:
	@echo "🎮 Starting competition bot..."
	@if [ ! -f "models/best_model.pt" ]; then \
		echo "❌ Model not found. Please train first:"; \
		echo "   make quick    (for quick baseline)"; \
		echo "   make train    (for full training)"; \
		exit 1; \
	fi
	uv run python competition_bot.py

setup:
	@echo "📋 Interactive setup..."
	uv run python setup_and_run.py

clean:
	@echo "🧹 Cleaning temporary files..."
	rm -f temp_challenge.wav
	rm -f drone_acoustics_train_val_data.zip
	@echo "✅ Cleaned"

status:
	@echo "📊 Current Status:"
	@echo ""
	@if [ -d ".venv" ]; then \
		echo "✅ Virtual environment: Installed"; \
	else \
		echo "❌ Virtual environment: Not found (run: make install)"; \
	fi
	@if [ -d "data/raw/train" ]; then \
		echo "✅ Training dataset: Found"; \
	else \
		echo "❌ Training dataset: Not found (run: make setup)"; \
	fi
	@if [ -d "data/raw/val" ]; then \
		echo "✅ Validation dataset: Found"; \
	else \
		echo "❌ Validation dataset: Not found (run: make setup)"; \
	fi
	@if [ -f "models/best_model.pt" ]; then \
		echo "✅ Model: Found"; \
	else \
		echo "❌ Model: Not found (run: make quick or make train)"; \
	fi
	@echo ""
	@echo "Ready to compete: "
	@if [ -f "models/best_model.pt" ]; then \
		echo "✅ YES - Run: make compete"; \
	else \
		echo "❌ NO - Run: make quick (for quick test) or make train (for full competition)"; \
	fi

