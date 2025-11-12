#!/bin/bash

# build_env.sh
# Script to set up the Python environment using uv

set -e  # Exit immediately if a command exits with a non-zero status

echo "📦 Setting up Python environment with uv..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv could not be found. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found in the current directory."
    exit 1
fi

# Install dependencies using uv
uv pip install -r requirements.txt

echo "✅ Environment setup complete!"