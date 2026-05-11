"""
Flask API Entry Point
This script acts as the bridge between the React Frontend and the Python ML logic.
It handles incoming JSON requests and returns model predictions.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS  # Allows React (Port 3000) to talk to Flask (Port 5000)
from predictor import FraudPredictor
import os

app = Flask(__name__)

# Enable CORS: Without this, your browser will block React from talking to Flask
# for security reasons. This 'whitelists' the frontend.
CORS(app)

# Initialize the Brain: Load models into memory once at startup
# This ensures predictions are nearly instantaneous
try:
    predictor = FraudPredictor()
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    predictor = None

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main API Endpoint for Fraud Scoring
    Expects: JSON object with 30 features (Time, V1-V28, Amount)
    Returns: JSON object with Decision, Probability, and Confidence Tiers
    """
    try:
        # 1. Capture the JSON data sent by the React frontend
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # 2. Basic Validation: Ensure all values are numeric
        # React sends strings from input fields; we convert them to floats here
        processed_input = {feature: float(value) for feature, value in data.items()}

        # 3. Run the ML Pipeline (Scaling -> Anomaly Scoring -> AutoGluon)
        # This calls the predict method in your predictor.py
        result = predictor.predict(processed_input)

        # 4. Return the result dictionary as a JSON response to React
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": f"Invalid input format: {ve}"}), 400
    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Status check to ensure the API is running."""
    return jsonify({"status": "active", "model_loaded": predictor is not None}), 200

if __name__ == '__main__':
    # Running on port 5000 by default
    app.run(debug=True, port=5000)
