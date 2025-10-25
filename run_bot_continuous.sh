#!/bin/bash
# Run the competition bot continuously

echo "🚀 Starting Competition Bot"
echo "================================"
echo ""
echo "Your credentials:"
echo "  Username: eshaank08"
echo "  Token: f276bbf9-e42b-452c-be54-eac3d4c6f0e3"
echo ""
echo "📊 Watch live at:"
echo "  https://edth.helsing.codes/static/index.html"
echo ""
echo "Press Ctrl+C to stop"
echo "================================"
echo ""

cd /Users/eshaan/Downloads/edth-munich-drone-acoustics
/Users/eshaan/.local/bin/uv run python competition_bot.py

