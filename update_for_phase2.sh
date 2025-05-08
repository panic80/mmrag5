#!/bin/bash
# Update script for Phase 2 improvements
# This script installs the additional dependencies required for Phase 2

set -e  # Exit immediately if a command exits with non-zero status

echo "===== RAG Pipeline Phase 2 Dependency Update ====="
echo "This script will install additional dependencies required for Phase 2 improvements."
echo ""

# Check for Python
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "Error: Python is not installed. Please install Python 3.8 or newer."
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# Check for pip
if ! $PYTHON -m pip --version &>/dev/null; then
    echo "Error: pip is not installed. Please install pip for Python."
    exit 1
fi

echo "Using pip: $($PYTHON -m pip --version)"
echo ""

# Install dependencies
echo "Installing Phase 2 dependencies..."
echo ""

# Core requirements
echo "1. Installing core dependencies..."
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -r requirements.txt

# Phase 2 specific requirements
echo ""
echo "2. Installing Phase 2 specific dependencies..."
$PYTHON -m pip install scikit-learn==1.4.0 numpy>=1.26.0 sentence-transformers>=2.2.2

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Dependencies successfully installed!"
else
    echo ""
    echo "❌ Error: Failed to install dependencies. Please check error messages above."
    exit 1
fi

# Update requirements.txt
echo ""
echo "3. Updating requirements.txt with new dependencies..."
if ! grep -q "scikit-learn==1.4.0" requirements.txt; then
    echo "scikit-learn==1.4.0" >> requirements.txt
    echo "Added scikit-learn to requirements.txt"
fi

if ! grep -q "sentence-transformers" requirements.txt; then
    echo "sentence-transformers>=2.2.2" >> requirements.txt
    echo "Added sentence-transformers to requirements.txt"
fi

echo ""
echo "===== Phase 2 Update Complete ====="
echo ""
echo "You can now run the demo script to see the new features:"
echo "python demo_phase2_improvements.py --collection your_collection --query \"your query\""
echo ""
echo "For more information about the Phase 2 improvements, see:"
echo "docs/PHASE2_IMPROVEMENTS.md"