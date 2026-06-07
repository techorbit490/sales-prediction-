#!/bin/bash
echo "========================================"
echo "  Sales Prediction App - Starting..."
echo "========================================"
echo ""

cd "$(dirname "$0")/backend"

echo "[1/2] Installing requirements..."
pip install flask flask-cors scikit-learn==1.3.2 pandas numpy joblib -q

echo ""
echo "[2/2] Starting Flask server..."
echo ""
echo "  App chalega at: http://localhost:5000"
echo "  Browser mein ye URL kholo!"
echo ""
echo "  (Band karne ke liye Ctrl+C dabao)"
echo "========================================"

python app.py
