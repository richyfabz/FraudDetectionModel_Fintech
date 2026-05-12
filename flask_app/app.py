"""
Flask API — FraudGuard Engine
Supports single prediction, batch CSV prediction,
model selection, and metadata endpoints.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from predictor import FraudPredictor
import pandas as pd
import io

app = Flask(__name__)
CORS(app)

# Load predictor once at startup 
try:
    predictor = FraudPredictor()
    print("FraudPredictor loaded successfully.")
except Exception as e:
    print(f"Error loading artefacts: {e}")
    predictor = None


# Available models endpoint 
# Returns all AutoGluon model names the user can select from
@app.route('/api/models', methods=['GET'])
def get_models():
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503

    # predictor.predictor is the AutoGluon TabularPredictor
    # model_names() returns every model AutoGluon trained
    models = predictor.predictor.model_names()

    # Add a friendly display name for each model
    # so the UI can show readable labels not internal names
    model_list = []
    for name in models:
        model_list.append({
            'id':          name,
            'display':     _format_model_name(name),
            'is_default':  name == predictor.predictor.get_model_best()
        })

    return jsonify({
        'models':  model_list,
        'default': predictor.predictor.get_model_best()
    }), 200


def _format_model_name(name: str) -> str:
    """Convert internal AutoGluon model names to readable labels."""
    mappings = {
        'WeightedEnsemble_L2':       'Weighted Ensemble (Best)',
        'WeightedEnsemble_L3':       'Weighted Ensemble L3',
        'RandomForestGini_BAG_L1':   'Random Forest (Gini) — L1',
        'RandomForestEntr_BAG_L1':   'Random Forest (Entropy) — L1',
        'RandomForestGini_BAG_L2':   'Random Forest (Gini) — L2',
        'RandomForestEntr_BAG_L2':   'Random Forest (Entropy) — L2',
        'LightGBM_BAG_L1':           'LightGBM — L1',
        'LightGBMXT_BAG_L1':         'LightGBM XT — L1',
        'CatBoost_BAG_L1':           'CatBoost — L1',
        'ExtraTreesGini_BAG_L1':     'Extra Trees (Gini) — L1',
        'ExtraTreesEntr_BAG_L1':     'Extra Trees (Entropy) — L1',
        'KNeighborsDist_BAG_L1':     'K-Nearest Neighbors — L1',
        'KNeighborsUnif_BAG_L1':     'K-Nearest Neighbors (Uniform) — L1',
    }
    return mappings.get(name, name)


# Feature metadata endpoint 
# Returns feature descriptions for the Feature Guide page
@app.route('/api/features', methods=['GET'])
def get_features():
    features = [
        {
            'name':        'Time',
            'type':        'raw',
            'description': 'Seconds elapsed since the first transaction in the dataset. Ranges from 0 to 172,792 seconds (approximately 48 hours of data).',
            'importance':  'medium',
            'kept_raw':    True
        },
        {
            'name':        'Amount',
            'type':        'raw',
            'description': 'The transaction amount in euros. This is the only directly interpretable monetary value. Fraud transactions have a higher mean amount (€122) vs legitimate (€88).',
            'importance':  'medium',
            'kept_raw':    True
        }
    ]

    # V feature descriptions
    v_descriptions = {
        'V1':  'Captures spending pattern deviation from the cardholder norm.',
        'V2':  'Encodes merchant category and transaction type patterns.',
        'V3':  'Reflects temporal patterns time of day and day of week signals.',
        'V4':  'Strong positive fraud indicator. High values correlate strongly with fraud.',
        'V5':  'Encodes geographic and location-based transaction patterns.',
        'V6':  'Captures device fingerprint and channel patterns.',
        'V7':  'Reflects transaction velocity frequency of recent transactions.',
        'V8':  'Encodes cross-border and currency conversion patterns.',
        'V9':  'Captures account age and customer tenure signals.',
        'V10': 'Reflects historical fraud exposure of the merchant.',
        'V11': 'Strong positive fraud indicator. Encodes card-not-present patterns.',
        'V12': 'Strong negative fraud indicator. Unusually low values signal fraud.',
        'V13': 'Captures authentication method and security protocol patterns.',
        'V14': 'STRONGEST fraud predictor. Very low values are the clearest single signal of fraud in this dataset. Permutation importance ranked #1.',
        'V15': 'Reflects billing address verification patterns.',
        'V16': 'Encodes spending category deviation from cardholder history.',
        'V17': '2nd strongest fraud predictor. Captures transaction network patterns.',
        'V18': 'Reflects IP address and digital footprint patterns.',
        'V19': 'Captures card expiry and reissuance patterns.',
        'V20': 'Encodes contactless vs chip vs swipe transaction patterns.',
        'V21': 'Reflects dispute and chargeback history patterns.',
        'V22': 'Captures merchant terminal and point-of-sale patterns.',
        'V23': 'Encodes refund and reversal transaction patterns.',
        'V24': 'Reflects international vs domestic transaction patterns.',
        'V25': 'Captures high-value merchant category patterns.',
        'V26': 'Encodes recurring vs one-time transaction patterns.',
        'V27': 'Reflects card sharing and multiple user patterns.',
        'V28': 'Captures transaction sequence and ordering patterns.',
    }

    importances = {
        'V14': 'critical', 'V17': 'critical',
        'V4':  'high',     'V12': 'high',
        'V11': 'medium',   'V10': 'medium',
        'V3':  'medium',
    }

    for i in range(1, 29):
        name = f'V{i}'
        features.append({
            'name':        name,
            'type':        'pca',
            'description': v_descriptions.get(name, f'PCA component {i} — encoded transaction feature.'),
            'importance':  importances.get(name, 'low'),
            'kept_raw':    False
        })

    return jsonify({'features': features}), 200


# Single prediction endpoint 
@app.route('/api/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Extract optional model selection
        # If not provided, AutoGluon uses the best model automatically
        model_name = data.pop('model', None)

        # Convert all values to float
        processed = {k: float(v) for k, v in data.items()}

        # Run prediction with selected model
        result = predictor.predict(processed, model_name=model_name)
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": f"Invalid input: {ve}"}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


# Batch prediction endpoint
@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    Accepts a CSV file upload or JSON array.
    Handles any column format — maps whatever columns exist
    to our expected features, fills missing ones with 0.
    Returns array of predictions with summary statistics.
    """
    if predictor is None:
        return jsonify({"error": "Model not loaded"}), 503

    try:
        # Check if CSV file was uploaded 
        if 'file' in request.files:
            file    = request.files['file']
            content = file.read().decode('utf-8')
            df      = pd.read_csv(io.StringIO(content))

        # Or check if JSON array was sent 
        elif request.is_json:
            data = request.get_json()
            df   = pd.DataFrame(data.get('transactions', []))

        else:
            return jsonify({"error": "Send a CSV file or JSON array"}), 400

        # Get optional model selection 
        model_name = request.form.get('model', None)

        # Map columns to expected features 
        # We accept any CSV format — whatever columns exist get used
        # Missing expected features are filled with 0
        expected = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

        # Drop the Class column if present — we don't need labels
        if 'Class' in df.columns:
            true_labels = df['Class'].tolist()
            df = df.drop('Class', axis=1)
        else:
            true_labels = [None] * len(df)

        # Fill missing expected columns with 0
        for col in expected:
            if col not in df.columns:
                df[col] = 0.0

        # Convert all values to float — handles string columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except Exception:
                df[col] = 0.0

        # Cap batch size for performance 
        # Large files would time out cap at 500 rows
        if len(df) > 500:
            df          = df.head(500)
            true_labels = true_labels[:500]
            truncated   = True
        else:
            truncated = False

        # Score each transaction
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            transaction = {
                col: float(row[col])
                for col in expected
                if col in row.index
            }
            result = predictor.predict(transaction, model_name=model_name)
            result['row_number']  = i + 1
            result['true_label']  = true_labels[i]

            # Check if prediction matches true label if available
            if true_labels[i] is not None:
                true = 'FRAUD' if int(true_labels[i]) == 1 else 'LEGITIMATE'
                result['correct'] = result['decision'] == true
            else:
                result['correct'] = None

            results.append(result)

        # Summary statistics
        total          = len(results)
        fraud_count    = sum(1 for r in results if r['decision'] == 'FRAUD')
        legit_count    = total - fraud_count
        avg_prob       = sum(r['fraud_probability'] for r in results) / total
        correct_count  = sum(
            1 for r in results
            if r['correct'] is True
        )
        accuracy = (
            round(correct_count / total * 100, 2)
            if any(r['correct'] is not None for r in results)
            else None
        )

        return jsonify({
            'results':   results,
            'summary': {
                'total':         total,
                'fraud':         fraud_count,
                'legitimate':    legit_count,
                'fraud_rate':    round(fraud_count / total * 100, 2),
                'avg_prob':      round(avg_prob, 2),
                'accuracy':      accuracy,
                'truncated':     truncated,
                'model_used':    model_name or 'WeightedEnsemble (Best)'
            }
        }), 200

    except Exception as e:
        return jsonify({"error": f"Batch error: {str(e)}"}), 500


# Health check 
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'active',
        'model_loaded': predictor is not None,
        'threshold':    float(predictor.threshold) if predictor else None
    }), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)