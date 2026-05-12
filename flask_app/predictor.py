# flask_app/predictor.py
# Fraud prediction engine for FraudGuard Flask application
# Loads all trained artefacts once at startup and exposes:
# - predict()     → single transaction scoring
# - get_models()  → list of available AutoGluon models
# Both are called by app.py routes and never retrain anything.

import numpy as np
import pandas as pd
import joblib
import os
import sys

# Add project root to Python path 
# predictor.py lives inside flask_app/ but models/ is one level up
# This tells Python where to find the models/ directory
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from autogluon.tabular import TabularPredictor
from sklearn.ensemble import IsolationForest

# Absolute paths — work regardless of where Flask is launched
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
AG_PATH    = os.path.join(MODELS_DIR, 'autogluon')


class FraudPredictor:
    """
    Loads all trained artefacts once at startup and runs the full
    hybrid prediction pipeline for every incoming transaction.

    Pipeline:
        1. Safe feature mapping  — fill missing columns with 0
        2. StandardScaler        — normalise to training scale
        3. Isolation Forest      — compute anomaly score
        4. Normalise score       — map raw score to [0, 1]
        5. Add anomaly_score     — append as 31st feature
        6. AutoGluon predict     — weighted ensemble or chosen model
        7. Apply threshold       — convert probability to decision
        8. Assign confidence tier
        9. Return structured dict
    """

    def __init__(self):
        print("Loading artefacts...")

        # StandardScaler
        # Fitted on training data only in Notebook 2
        # Must be applied to every transaction before scoring
        self.scaler = joblib.load(
            os.path.join(MODELS_DIR, 'scaler.joblib')
        )

        # Isolation Forest
        # Trained on 226,602 legitimate transactions only
        # Generates an unsupervised anomaly score per transaction
        self.isolation_forest = joblib.load(
            os.path.join(MODELS_DIR, 'isolation_forest.joblib')
        )

        # Normalisation parameters
        # global_min and global_max computed across training + test
        # anomaly scores in Notebook 3. Required to map new scores
        # to the same [0,1] scale AutoGluon was trained with.
        norm_params     = joblib.load(
            os.path.join(MODELS_DIR, 'anomaly_norm_params.joblib')
        )
        self.global_min = norm_params['global_min']
        self.global_max = norm_params['global_max']

        # Decision threshold 
        # 0.397 — the value that maximised F1-Score during
        # threshold tuning in Notebook 3
        self.threshold = joblib.load(
            os.path.join(MODELS_DIR, 'threshold.joblib')
        )

        # AutoGluon predictor 
        # Loads the entire AutoGluon model directory — all base
        # models and the weighted ensemble stack
        self.predictor = TabularPredictor.load(AG_PATH)

        # Expected feature order 
        # Scaler and AutoGluon both require this exact column order
        # Any mismatch silently corrupts predictions
        self.feature_names = (
            ['Time'] +
            [f'V{i}' for i in range(1, 29)] +
            ['Amount']
        )

        print("All artefacts loaded.")
        print(f"Threshold:  {self.threshold}")
        print(f"Features:   {len(self.feature_names)} + anomaly_score")


    def get_models(self) -> list:
        """
        Returns all AutoGluon model names from the leaderboard.
        Called by the /api/models endpoint to populate the
        ModelSelector dropdown in the React frontend.

        Returns:
            List of model name strings
        """
        # leaderboard() returns a DataFrame we extract the
        # 'model' column and convert to a plain Python list
        lb = self.predictor.leaderboard(silent=True)
        return lb['model'].tolist()


    def predict(
        self,
        raw_input: dict,
        model_name: str = None
    ) -> dict:
        """
        Run the full prediction pipeline for one transaction.

        Args:
            raw_input:  Dict of feature_name → value.
                        Can be missing fields — they default to 0.
            model_name: Optional AutoGluon model name.
                        None uses the best model (WeightedEnsemble).

        Returns:
            Dict with decision, probabilities, tier, and raw input.
        """

        # Step 1 — Safe feature mapping 
        # .get(feat, 0.0) fills any missing column with 0
        # This prevents crashes when a CSV is missing V28 etc.
        # We then enforce the exact column order the scaler expects
        safe_input = {
            feat: float(raw_input.get(feat, 0.0))
            for feat in self.feature_names
        }
        raw_df = pd.DataFrame([safe_input])[self.feature_names]

        # Step 2 — Scale features
        # transform() applies saved mean and std from training
        # We never call fit_transform() — that would refit on one
        # transaction which is meaningless and causes data leakage
        scaled_array = self.scaler.transform(raw_df)
        scaled_df    = pd.DataFrame(
            scaled_array,
            columns=self.feature_names
        )

        # Step 3 — Isolation Forest anomaly score 
        # decision_function returns a raw score:
        #   negative = anomalous (likely fraud)
        #   positive = normal (likely legitimate)
        raw_score = self.isolation_forest.decision_function(
            scaled_df
        )[0]

        # Step 4 — Normalise anomaly score to [0, 1]
        # Flip sign so higher = more fraudulent
        # Then normalise using the same global min/max from training
        # so the scale is consistent with what AutoGluon learned
        flipped         = -raw_score
        global_min_flip = -self.global_max
        global_max_flip = -self.global_min
        anomaly_score   = (
            (flipped - global_min_flip) /
            (global_max_flip - global_min_flip)
        )

        # Step 5 — Add anomaly score as 31st feature 
        # AutoGluon was trained with anomaly_score included
        # It must appear in every inference call in the same position
        scaled_df['anomaly_score'] = anomaly_score

        # Step 6 — AutoGluon prediction 
        # If model_name is provided → score with that specific model
        # If None → AutoGluon uses WeightedEnsemble automatically
        # predict_proba returns DataFrame: col 0 = P(legit), col 1 = P(fraud)
        if model_name:
            proba_df = self.predictor.predict_proba(
                scaled_df,
                model=model_name
            )
        else:
            proba_df = self.predictor.predict_proba(scaled_df)

        fraud_prob = float(proba_df[1].iloc[0])

        # Step 7 — Apply threshold 
        # Threshold was tuned to maximise F1-Score in Notebook 3
        # Above threshold → FRAUD, below → LEGITIMATE
        decision = 'FRAUD' if fraud_prob >= self.threshold else 'LEGITIMATE'

        # Step 8 — Confidence tier
        confidence_tier = self._get_confidence_tier(
            fraud_prob, decision
        )

        # Step 9 — Return structured result 
        return {
            'decision':          decision,
            'fraud_probability': round(fraud_prob * 100, 2),
            'confidence_tier':   confidence_tier,
            'threshold_used':    round(float(self.threshold) * 100, 2),
            'anomaly_score':     round(float(anomaly_score) * 100, 2),
            'model_used':        model_name or 'WeightedEnsemble (Best)',
            'raw_input':         raw_input
        }


    def _get_confidence_tier(
        self,
        prob: float,
        decision: str
    ) -> str:
        """
        Converts raw probability to a human-readable confidence tier.

        FRAUD:
            HIGH   → prob >= 0.90  (act immediately)
            MEDIUM → prob >= threshold but < 0.90 (investigate)

        LEGITIMATE:
            CLEAR  → prob <= 0.30  (no action needed)
            REVIEW → prob > 0.30 but < threshold (monitor)
        """
        if decision == 'FRAUD':
            return 'HIGH'   if prob >= 0.90 else 'MEDIUM'
        else:
            return 'CLEAR'  if prob <= 0.30 else 'REVIEW'


#  Singleton instantiation 
# Python caches imports this runs exactly once when Flask starts
# Every request reuses the same in-memory FraudPredictor object
# Loading AutoGluon per-request would add 10+ seconds per call
predictor = FraudPredictor()