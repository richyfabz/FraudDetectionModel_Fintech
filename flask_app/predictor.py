# flask_app/predictor.py
# Fraud prediction logic for Flask application
# This file loads all trained artefacts once when the Flask app
# starts and exposes a single predict() function that the routes
# in app.py call for every form submission

import numpy as np
import pandas as pd
import joblib
import os
import sys

# Add project root to Python path 
# Flask runs from inside flask_app/ but our models/ directory
# is one level up at the project root. This line tells Python
# where to look when resolving file paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# AutoGluon import
# We only import TabularPredictor the class that loads and runs
# the trained AutoGluon ensemble. We don't need to retrain anything.
from autogluon.tabular import TabularPredictor

# IsolationForest import
# Needed to load the saved isolation_forest object from joblib
# Without this import joblib cannot deserialise the object correctly
from sklearn.ensemble import IsolationForest


# Define paths relative to project root
# os.path.abspath(__file__) gets the absolute path of predictor.py
# os.path.dirname() twice walks up two levels to the project root
# This makes the app work regardless of where it's launched from
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
AG_PATH     = os.path.join(MODELS_DIR, 'autogluon')


class FraudPredictor:
    """
    Loads all trained artefacts once at startup and runs the full
    prediction pipeline for every incoming transaction.

    Pipeline:
        1. Scale raw features using saved StandardScaler
        2. Compute Isolation Forest anomaly score
        3. Add anomaly score to feature matrix
        4. AutoGluon predict + predict_proba
        5. Apply tuned threshold
        6. Return structured result dict
    """

    def __init__(self):
        print("Loading artefacts...")

        # Load StandardScaler
        # This scaler was fitted on training data only in Notebook 2
        # It must be applied to every incoming transaction using the
        # exact same mean and std values learned during training
        self.scaler = joblib.load(
            os.path.join(MODELS_DIR, 'scaler.joblib')
        )

        # Load Isolation Forest
        # The trained IF model that generates anomaly scores
        # It was trained on 226,602 legitimate transactions only
        self.isolation_forest = joblib.load(
            os.path.join(MODELS_DIR, 'isolation_forest.joblib')
        )

        # Load normalisation parameters 
        # global_min and global_max were computed across both
        # training and test anomaly scores in Notebook 3
        # We need them to normalise new anomaly scores to [0,1]
        # using the exact same scale as during training
        norm_params      = joblib.load(
            os.path.join(MODELS_DIR, 'anomaly_norm_params.joblib')
        )
        self.global_min  = norm_params['global_min']
        self.global_max  = norm_params['global_max']

        # Load decision threshold
        # 0.41 the threshold that maximised F1-Score during
        # threshold tuning in Notebook 3
        self.threshold = joblib.load(
            os.path.join(MODELS_DIR, 'threshold.joblib')
        )

        # Load AutoGluon predictor 
        # TabularPredictor.load() reads the entire AutoGluon model
        # directory all base models and the weighted ensemble
        # This is AutoGluon's own persistence format, not joblib
        self.predictor = TabularPredictor.load(AG_PATH)

        # Define expected feature order
        # The scaler and AutoGluon both expect features in this
        # exact order. Any mismatch silently corrupts predictions.
        self.feature_names = (
            ['Time'] +
            [f'V{i}' for i in range(1, 29)] +
            ['Amount']
        )

        print(f"All artefacts loaded.")
        print(f"Threshold: {self.threshold}")
        print(f"Features:  {len(self.feature_names)} + anomaly_score")


    def predict(self, raw_input: dict, model_name: str = None) -> dict:
        """
        Run the full prediction pipeline for one transaction.

        Args:
            raw_input:  dict with keys matching feature_names
        model_name: optional AutoGluon model to use.
                    None uses the best model automatically.
        Returns:
            dict with decision, probability, confidence tier,
            threshold used, and feature values for display
        """

        # step 1 — Build one-row DataFrame
        # pd.DataFrame([raw_input]) converts the dict to a single
        # row DataFrame. The [self.feature_names] ensures columns
        # are in the exact order the scaler expects
        raw_df = pd.DataFrame([raw_input])[self.feature_names]

        # step 2 — Scale features using saved StandardScaler
        # transform() applies the saved scaler parameters
        # We never call fit_transform() here that would refit
        # the scaler on a single transaction which is meaningless
        scaled_array = self.scaler.transform(raw_df)
        scaled_df    = pd.DataFrame(
            scaled_array,
            columns=self.feature_names
        )

        # step 3 — Compute Isolation Forest anomaly score
        # decision_function() returns a raw anomaly score
        # Negative = anomalous (likely fraud)
        # Positive = normal (likely legitimate)
        raw_score = self.isolation_forest.decision_function(scaled_df)[0]

        # step 4 — Normalise anomaly score to [0, 1]
        # We flip the sign so higher score = more fraudulent
        # Then normalise using the global min/max from training
        # so the scale is consistent with what AutoGluon learned
        flipped          = -raw_score
        global_min_flip  = -self.global_max
        global_max_flip  = -self.global_min
        anomaly_score    = (flipped - global_min_flip) / \
                           (global_max_flip - global_min_flip)

        # step 5 — Add anomaly score to feature matrix 
        # AutoGluon was trained with anomaly_score as the 31st
        # feature. We must add it in the same position here.
        scaled_df['anomaly_score'] = anomaly_score

        # step 6 — AutoGluon prediction 
        # If model_name is provided, score with that specific model
    # If None, AutoGluon uses WeightedEnsemble automatically
        if model_name:
            proba_df = self.predictor.predict_proba(
            scaled_df, model=model_name
         )
        else:
            proba_df = self.predictor.predict_proba(scaled_df)

        fraud_prob = float(proba_df[1].iloc[0])

        # step 7 — Apply threshold 
        # If fraud probability >= threshold → flag as FRAUD
        # If fraud probability < threshold  → clear as LEGITIMATE
        decision = 'FRAUD' if fraud_prob >= self.threshold else 'LEGITIMATE'

        # step 8 — Assign confidence tier 
        confidence_tier = self._get_confidence_tier(
            fraud_prob, decision
        )

        # step 9 — Return structured result 
        return {
            'decision':          decision,
            'fraud_probability': round(fraud_prob * 100, 2),
            'confidence_tier':   confidence_tier,
            'threshold_used':    round(self.threshold * 100, 2),
            'anomaly_score':     round(float(anomaly_score) * 100, 2),
            'raw_input':         raw_input
        }


    def _get_confidence_tier(self, prob: float, decision: str) -> str:
        """
        Translate raw fraud probability into a human-readable
        confidence tier for the Flask result page.

        FRAUD tiers:
            HIGH   — model is very confident (prob >= 0.90)
            MEDIUM — model suspects fraud (prob >= threshold)

        LEGITIMATE tiers:
            CLEAR  — model is very confident (prob <= 0.30)
            REVIEW — low fraud probability but worth monitoring
        """
        if decision == 'FRAUD':
            return 'HIGH'   if prob >= 0.90 else 'MEDIUM'
        else:
            return 'CLEAR'  if prob <= 0.30 else 'REVIEW'


# Instantiate once at module level
# Python imports are cached this block runs exactly once when
# Flask first imports predictor.py. Every subsequent request
# reuses the same FraudPredictor instance already in memory.
# Loading a full AutoGluon ensemble takes several seconds
# doing it per-request would make the app unusably slow.
predictor = FraudPredictor()