#!/bin/bash

# EEG Lab Launch Script
echo "========================================="
echo "     EEG Lab - Research Platform"
echo "========================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements if needed
if [ ! -f "venv/installed" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    touch venv/installed
    echo "Requirements installed successfully!"
fi

# Create necessary directories
mkdir -p data/uploads data/datasets notebooks models results

echo "Starting EEG Lab application..."
echo "Access the application at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo "========================================="

# Start the Flask application
python app.py
