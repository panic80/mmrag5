#!/bin/bash
# Update script for Phase 3 of the RAG Retrieval Pipeline

set -e

echo "===== RAG Retrieval Pipeline - Phase 3 Update ====="
echo "This script will update the project with Phase 3 improvements:"
echo "- Microservices Architecture"
echo "- ColBERT Implementation"
echo "- SPLADE Integration"
echo "- Dynamic Parameter Tuning"
echo "- Automated Strategy Selection"
echo ""

# Install required dependencies
echo "Installing dependencies from requirements_phase3.txt..."
pip install -r requirements_phase3.txt

# Create necessary directories
echo "Creating directories for indices and models..."
mkdir -p bm25_indices
mkdir -p splade_indices
mkdir -p parameter_models
mkdir -p strategy_models
mkdir -p colbert_data

# Set permissions for the update script and demo script
echo "Setting execution permissions for scripts..."
chmod +x demo_phase3.py

# Test imports
echo "Testing imports to verify installation..."
python -c "import torch; import transformers; import fastapi; import uvicorn; import redis; import sklearn.ensemble; print('All imports successful!')"

echo ""
echo "===== Phase 3 Update Complete ====="
echo ""
echo "To run the demonstration:"
echo "1. Start the microservices: docker-compose -f docker-compose-phase3.yml up"
echo "2. Run the demo script: python demo_phase3.py"
echo ""
echo "Individual services can be started manually with:"
echo "python -m uvicorn microservices.query_service:create_fastapi_app --host 0.0.0.0 --port 8004 --factory"
echo ""
echo "Enjoy the advanced RAG capabilities!"