# api/predictor.py
# Prediction logic isolated from API routing 
import numpy as np
import pandas as pd
import shap
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import engineer_features, get_feature_names
from src.preprocess import scale_single
from src.evaluate import apply_threshold, format_shap_explanation


class FraudPredictor:
    """
    Encapsulates all prediction logic — artefact loading,
    feature engineering, scaling, scoring, and explanation.

    Loaded once at API startup and reused for every request.
    """

    def __init__(self, models_dir: str = None):
        if models_dir is None:
            models_dir = os.path.join(
                os.path.dirname(__file__), '..', 'models'
            )

        # Load artefacts 
        self.model     = joblib.load(f'{models_dir}/ensemble_model.joblib')
        self.scaler    = joblib.load(f'{models_dir}/scaler.joblib')
        self.threshold = joblib.load(f'{models_dir}/threshold.joblib')

        # SHAP explainer on XGBoost base model 
        xgb_model      = self.model.named_estimators_['xgb']
        self.explainer  = shap.TreeExplainer(xgb_model)
        self.feature_names = get_feature_names()

        print(f"FraudPredictor ready — threshold: {self.threshold}")

    def predict(self, raw: dict) -> dict:
        """
        Run the full prediction pipeline for one transaction.

        Args:
            raw: Dict from parsed TransactionRequest

        Returns:
            Dict matching TransactionResponse schema
        """
        # Feature engineering 
        df         = pd.DataFrame([raw])
        engineered = engineer_features(df)

        # Scale 
        scaled = scale_single(
            self.scaler,
            engineered.iloc[0].to_dict(),
            self.feature_names
        )

        #  Prediction and explanation
        fraud_prob = float(self.model.predict_proba(scaled)[0, 1])
        prediction = apply_threshold(np.array([fraud_prob]), self.threshold)[0]
        decision   = 'FRAUD' if prediction == 1 else 'LEGITIMATE'

        # SHAP explanation 
        shap_values  = self.explainer.shap_values(scaled)
        top_features = format_shap_explanation(
            shap_values[0], self.feature_names, top_n=10
        )

        # Confidence tier logic 
        confidence_tier = self._get_confidence_tier(fraud_prob, decision)

        return {
            'transaction_id':    None,
            'decision':          decision,
            'fraud_probability': round(fraud_prob, 4),
            'confidence_tier':   confidence_tier,
            'threshold_used':    self.threshold,
            'top_features':      top_features
        }

    def _get_confidence_tier(self, prob: float, decision: str) -> str:
        if decision == 'FRAUD':
            return 'HIGH'   if prob >= 0.90 else 'MEDIUM'
        else:
            return 'CLEAR'  if prob <= 0.30 else 'REVIEW'