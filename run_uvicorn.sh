#!/bin/bash
# run_uvicorn.sh - Run FastAPI app with Uvicorn using mock KB
# Usage: ./run_uvicorn.sh [port]

PORT=${1:-8000}
echo "Starting Uvicorn on port $PORT with MOCK KB..."

# Kill any existing uvicorn processes
pkill -f "uvicorn main:app" || true
sleep 1

# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Set environment variables
export ENVIRONMENT=development

# Start uvicorn with hot reload for the app and subdirectories
uvicorn chatbot_api:app \
  --reload \
  --host 0.0.0.0 \
  --port $PORT \
  --log-level debug

echo "Uvicorn started on port $PORT"