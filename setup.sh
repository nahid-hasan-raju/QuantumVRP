#!/bin/bash
# Setup script for macOS/Linux
set -e

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo ""
echo "Setup complete. To get started:"
echo "  source venv/bin/activate"
echo "  python run.py"
