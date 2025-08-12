#!/bin/bash
set -e
echo "🚀 Installing AI Science Platform..."

# Install dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# Create directories
mkdir -p logs data backups

echo "✅ Installation completed!"
