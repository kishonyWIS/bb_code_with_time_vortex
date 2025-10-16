#!/bin/bash
# Script to properly activate the Python 3.13 virtual environment
# This ensures the venv's Python takes precedence over pyenv

cd /Users/giladkishony/PycharmProjects/bb_code_with_time_vortex
source venv/bin/activate
export PATH="/Users/giladkishony/PycharmProjects/bb_code_with_time_vortex/venv/bin:$PATH"

echo "✅ Virtual environment activated with Python 3.13"
echo "Python version: $(python --version)"
echo "Python location: $(which python)"
echo ""
echo "To run your project:"
echo "  python examples.py"
echo "  python test_framework.py"
echo "  # or any other Python script"
