"""
Flask API Entry Point
This script acts as the bridge between the React Frontend
and the Python ML logic. It handles incoming JSON requests
and returns model predictions.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import FraudPredictor
import os

app = Flask(__name__)

# ── Enable CORS ───────────────────────────────────────────────────
# Without this the browser blocks React (port 3000) from calling
# Flask (port 5000) due to the same-origin security policy
CORS(app)

# ── Load models once at startup ───────────────────────────────────
# FraudPredictor loads AutoGluon, IsolationForest, and StandardScaler
# into memory. Doing this once at startup means every prediction
# request is nearly instantaneous — no repeated disk reads
try:
    predictor = FraudPredictor()
    print("FraudPredictor loaded successfully.")
except Exception as e:
    print(f"Error loading model artefacts: {e}")
    predictor = None


# ── Main prediction endpoint ──────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Receives a JSON object with 30 transaction features from React.
    Runs the full ML pipeline and returns the fraud decision.

    Expected input:
        {Time: float, V1-V28: float, Amount: float}

    Returns:
        {decision, fraud_probability, confidence_tier,
         threshold_used, anomaly_score, raw_input}
    """
    # ── Guard clause — model must be loaded ───────────────────────
    # If FraudPredictor failed to load at startup, we return a
    # meaningful error instead of a confusing AttributeError
    if predictor is None:
        return jsonify({
            "error": "Model not loaded. Check server logs."
        }), 503

    try:
        # ── Step 1 — Parse incoming JSON ──────────────────────────
        # request.get_json() reads the raw JSON body React sent
        # Returns None if the body is empty or malformed JSON
        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # ── Step 2 — Convert all values to float ──────────────────
        # React form inputs send strings even for number fields
        # The ML pipeline expects Python floats not strings
        # This dict comprehension converts every value in one pass
        processed_input = {
            feature: float(value)
            for feature, value in data.items()
        }

        # ── Step 3 — Run prediction pipeline ──────────────────────
        # predictor.predict() handles:
        # scaling → anomaly score → AutoGluon → threshold → tier
        result = predictor.predict(processed_input)

        # ── Step 4 — Return JSON response to React ────────────────
        # jsonify() converts the Python dict to a proper JSON
        # response with Content-Type: application/json header
        return jsonify(result), 200

    except ValueError as ve:
        # Handles cases like non-numeric strings slipping through
        return jsonify({"error": f"Invalid input format: {ve}"}), 400

    except Exception as e:
        # Catches any unexpected error in the pipeline
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


# Health check endpoint 
@app.route('/api/health', methods=['GET'])
def health():
    """
    Simple status check endpoint.
    React can ping this on startup to confirm Flask is running
    and the model loaded successfully before showing the form.
    """
    return jsonify({
        "status":       "active",
        "model_loaded": predictor is not None,
        "threshold":    float(predictor.threshold) if predictor else None
    }), 200


# Entry point
if __name__ == '__main__':
    # debug=True enables auto-reload on file save during development
    # Never use debug=True in production — it exposes the debugger
    app.run(debug=True, port=5000)