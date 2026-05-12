# flask_app/app.py
# FraudGuard Flask API 
# Bridge between the React frontend and the Python ML pipeline.
# Exposes four endpoints:
#   GET  /api/health          → server status check
#   GET  /api/models          → list of available AutoGluon models
#   GET  /api/features        → feature metadata for Feature Guide page
#   POST /api/predict         → single transaction scoring
#   POST /api/predict/batch   → CSV or JSON batch scoring

from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import FraudPredictor
import pandas as pd
import io

app = Flask(__name__)

# CORS — allow React (port 3000) to call Flask (port 5000) 
# Without this the browser blocks cross-origin requests
CORS(app)

# Load predictor once at startup, keep in memory for all requests 
# All AutoGluon models + scaler + IF load here
# Subsequent requests reuse this in-memory object
try:
    predictor = FraudPredictor()
    print("FraudPredictor ready.")
except Exception as e:
    print(f"Startup error: {e}")
    predictor = None


# Health check
@app.route('/api/health', methods=['GET'])
def health():
    """React pings this on load to confirm Flask is running."""
    return jsonify({
        'status':       'active',
        'model_loaded': predictor is not None,
        'threshold':    float(predictor.threshold) if predictor else None
    }), 200


# Available models 
@app.route('/api/models', methods=['GET'])
def list_models():
    """
    Returns all AutoGluon model names for the ModelSelector dropdown.
    Each entry has an id (internal name) and a display name.
    """
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503

    raw_models = predictor.get_models()

    # Friendly display names 
    # Map internal AutoGluon names to readable UI labels
    name_map = {
        'WeightedEnsemble_L2':     'Weighted Ensemble (Best)',
        'WeightedEnsemble_L3':     'Weighted Ensemble L3',
        'RandomForestGini_BAG_L1': 'Random Forest Gini — L1',
        'RandomForestEntr_BAG_L1': 'Random Forest Entropy — L1',
        'RandomForestGini_BAG_L2': 'Random Forest Gini — L2',
        'RandomForestEntr_BAG_L2': 'Random Forest Entropy — L2',
        'LightGBM_BAG_L1':         'LightGBM — L1',
        'LightGBMXT_BAG_L1':       'LightGBM XT — L1',
        'CatBoost_BAG_L1':         'CatBoost — L1',
        'ExtraTreesGini_BAG_L1':   'Extra Trees Gini — L1',
        'ExtraTreesEntr_BAG_L1':   'Extra Trees Entropy — L1',
        'KNeighborsDist_BAG_L1':   'K-Nearest Neighbors — L1',
        'KNeighborsUnif_BAG_L1':   'K-Nearest Neighbors Uniform — L1',
    }

    # In AutoGluon 1.0+, use model_best instead of get_model_best()
    best = predictor.predictor.model_best

    models = [
        {
            'id':         name,
            'display':    name_map.get(name, name),
            'is_default': name == best
        }
        for name in raw_models
    ]

    return jsonify({
        'models':  models,
        'default': best
    }), 200


# Feature metadata 
@app.route('/api/features', methods=['GET'])
def get_features():
    """
    Returns feature descriptions for the Feature Guide page.
    Explains what each V component represents and its importance.
    """
    features = [
        {
            'name':        'Time',
            'type':        'raw',
            'importance':  'medium',
            'kept_raw':    True,
            'description': (
                'Seconds elapsed since the first transaction in the '
                'dataset. Ranges from 0 to 172,792 seconds '
                '(approximately 48 hours of data). Kept raw because '
                'Isolation Forest ranked it 2nd most important for '
                'anomaly detection.'
            )
        },
        {
            'name':        'Amount',
            'type':        'raw',
            'importance':  'medium',
            'kept_raw':    True,
            'description': (
                'Transaction amount in euros. The only directly '
                'interpretable monetary value. Fraud transactions '
                'average €122 vs €88 for legitimate. Kept raw — '
                'dropping it caused the model to miss 4 out of 10 '
                'fraud cases in testing.'
            )
        }
    ]

    # V feature metadata
    v_meta = {
        'V1':  ('low',      'PCA component encoding spending pattern deviation from the cardholder norm.'),
        'V2':  ('low',      'PCA component encoding merchant category and transaction type patterns.'),
        'V3':  ('medium',   'PCA component reflecting temporal patterns — time of day and day of week signals.'),
        'V4':  ('high',     'Strong positive fraud indicator. High V4 values correlate strongly with fraudulent transactions.'),
        'V5':  ('low',      'PCA component encoding geographic and location-based transaction patterns.'),
        'V6':  ('low',      'PCA component capturing device fingerprint and payment channel patterns.'),
        'V7':  ('low',      'PCA component reflecting transaction velocity — frequency of recent transactions.'),
        'V8':  ('low',      'PCA component encoding cross-border and currency conversion patterns.'),
        'V9':  ('low',      'PCA component capturing account age and customer tenure signals.'),
        'V10': ('medium',   'PCA component reflecting historical fraud exposure patterns at the merchant level.'),
        'V11': ('medium',   'PCA component encoding card-not-present and online transaction patterns.'),
        'V12': ('high',     'Strong negative fraud indicator. Unusually low V12 values are a clear fraud signal.'),
        'V13': ('low',      'PCA component capturing authentication method and security protocol patterns.'),
        'V14': ('critical', 'STRONGEST fraud predictor. Very low V14 values are the clearest single fraud signal. Ranked #1 by both AutoGluon permutation importance and SHAP analysis.'),
        'V15': ('low',      'PCA component reflecting billing address verification and AVS patterns.'),
        'V16': ('low',      'PCA component encoding spending category deviation from cardholder history.'),
        'V17': ('critical', '2nd strongest fraud predictor. Captures transaction network anomaly patterns.'),
        'V18': ('low',      'PCA component reflecting IP address and digital footprint patterns.'),
        'V19': ('low',      'PCA component capturing card expiry and reissuance patterns.'),
        'V20': ('low',      'PCA component encoding contactless vs chip vs swipe transaction patterns.'),
        'V21': ('low',      'PCA component reflecting dispute and chargeback history patterns.'),
        'V22': ('low',      'PCA component capturing merchant terminal and point-of-sale patterns.'),
        'V23': ('low',      'PCA component encoding refund and reversal transaction patterns.'),
        'V24': ('low',      'PCA component reflecting international vs domestic transaction patterns.'),
        'V25': ('low',      'PCA component capturing high-value merchant category patterns.'),
        'V26': ('low',      'PCA component encoding recurring vs one-time transaction patterns.'),
        'V27': ('low',      'PCA component reflecting card sharing and multiple user patterns.'),
        'V28': ('low',      'PCA component capturing transaction sequence and ordering patterns.'),
    }

    for i in range(1, 29):
        name = f'V{i}'
        importance, description = v_meta.get(
            name, ('low', f'PCA component {i}.')
        )
        features.append({
            'name':        name,
            'type':        'pca',
            'importance':  importance,
            'kept_raw':    False,
            'description': description
        })

    return jsonify({'features': features}), 200


# Single transaction prediction
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Scores one transaction.
    Accepts optional 'selected_model' field to choose the algorithm.
    Missing feature columns are filled with 0 automatically.
    """
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract optional model selection
        # React sends this from the ModelSelector dropdown
        # None means use the best model (WeightedEnsemble)
        selected_model = data.pop('selected_model', None)

        # Convert all values to float
        # React form inputs are strings — pipeline needs floats
        processed = {k: float(v) for k, v in data.items()}

        # Run pipeline and return result
        result = predictor.predict(processed, model_name=selected_model)
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": f"Invalid input: {ve}"}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# Batch prediction for CSV or JSON array
@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Scores multiple transactions from a CSV file or JSON array.
    Accepts any column format — missing columns filled with 0.
    Capped at 500 rows to prevent browser timeout.
    """
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        # Parse input — CSV file or JSON array 
        if 'file' in request.files:
            # CSV file upload from drag-and-drop
            content = request.files['file'].read().decode('utf-8')
            df      = pd.read_csv(io.StringIO(content))
        elif request.is_json:
            # JSON array from programmatic call
            df = pd.DataFrame(
                request.get_json().get('transactions', [])
            )
        else:
            return jsonify({
                "error": "Send a CSV file or JSON array"
            }), 400

        # Extract optional model selection
        selected_model = request.form.get('selected_model', None)

        # Extract true labels if present
        # Dataset CSVs include a Class column (0=legit, 1=fraud)
        # We extract it to show whether predictions are correct
        if 'Class' in df.columns:
            true_labels = df['Class'].tolist()
            df = df.drop('Class', axis=1)
        else:
            true_labels = [None] * len(df)

        # Cap at 500 rows to prevent browser timeout
        truncated = len(df) > 500
        if truncated:
            df          = df.head(500)
            true_labels = true_labels[:500]

        # Convert all columns to numeric 
        # Handles mixed-type CSVs with string columns
        for col in df.columns:
            df[col] = pd.to_numeric(
                df[col], errors='coerce'
            ).fillna(0)

        # Score each row and collect results
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            # Convert row to dict — missing features filled to 0
            # inside predictor.predict() via the safe_input logic
            transaction = {
                col: float(row[col])
                for col in df.columns
            }
            result = predictor.predict(
                transaction,
                model_name=selected_model
            )

            # Attach row metadata ─ row number, true label, correctness
            result['row_number'] = i + 1
            result['true_label'] = true_labels[i]

            # Check correctness if true label is available
            if true_labels[i] is not None:
                true = (
                    'FRAUD' if int(true_labels[i]) == 1
                    else 'LEGITIMATE'
                )
                result['correct'] = result['decision'] == true
            else:
                result['correct'] = None

            results.append(result)

        # Summary statistics
        total       = len(results)
        fraud_count = sum(
            1 for r in results if r['decision'] == 'FRAUD'
        )
        correct_count = sum(
            1 for r in results if r['correct'] is True
        )
        has_labels = any(r['correct'] is not None for r in results)

        return jsonify({
            'results': results,
            'summary': {
                'total':      total,
                'fraud':      fraud_count,
                'legitimate': total - fraud_count,
                'fraud_rate': round(fraud_count / total * 100, 2),
                'avg_prob':   round(
                    sum(r['fraud_probability'] for r in results) / total,
                    2
                ),
                'accuracy':   round(
                    correct_count / total * 100, 2
                ) if has_labels else None,
                'truncated':  truncated,
                'model_used': selected_model or 'WeightedEnsemble (Best)'
            }
        }), 200

    except Exception as e:
        return jsonify({"error": f"Batch error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)