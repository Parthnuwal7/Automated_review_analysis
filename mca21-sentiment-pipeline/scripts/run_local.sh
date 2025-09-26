#!/bin/bash

# Local development script for MCA21 Sentiment Analysis Pipeline
# This script sets up the environment and runs the Streamlit application

set -e  # Exit on any error

echo "🚀 Starting MCA21 Sentiment Analysis Pipeline..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

# Install dependencies if requirements.txt is newer than last install
if [ requirements.txt -nt venv/pyvenv.cfg ] || [ ! -f venv/.requirements_installed ]; then
    echo "📋 Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    # Download spaCy model
    echo "📚 Downloading spaCy model..."
    python -m spacy download en_core_web_sm
    
    # Mark requirements as installed
    touch venv/.requirements_installed
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "🔐 Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data logs models/cache outputs

# Generate sample data if it doesn't exist
if [ ! -f "data/sample_reviews.csv" ]; then
    echo "📄 Generating sample data..."
    python tools/generate_sample_data.py
fi

# Check if port is available
PORT=${STREAMLIT_PORT:-8501}
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port $PORT is already in use. Trying port $((PORT+1))..."
    PORT=$((PORT+1))
fi

echo "🌐 Starting Streamlit app on port $PORT..."
echo "📊 Dashboard will be available at: http://localhost:$PORT"
echo "⏹️  Press Ctrl+C to stop the application"
echo ""

# Start the Streamlit application
streamlit run app/main.py \
    --server.port=$PORT \
    --server.address=localhost \
    --server.headless=false \
    --browser.serverAddress=localhost \
    --browser.serverPort=$PORT
