#!/bin/bash

# Script to start the Flask backend server

cd backend

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if model exists
if [ ! -f "model/attrition_model.cbm" ]; then
    echo "⚠️  WARNING: Model file not found!"
    echo "Please train and save the model first by running the notebook or save_model.py"
    echo ""
    read -p "Do you want to train the model now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python save_model.py
    else
        echo "Exiting. Please train the model first."
        exit 1
    fi
fi

# Start the server
echo "Starting Flask server on http://localhost:5001"
echo "API endpoint: http://localhost:5001/api/predict"
echo "Health check: http://localhost:5001/api/health"
python app.py

