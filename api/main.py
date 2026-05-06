# api/main.py
# Fraud Detection REST API 
import numpy as np
import joblib
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import engineer_features, get_feature_names
from src.preprocess import scale_single
from src.evaluate import apply_threshold, format_shap_explanation

import pandas as pd


# App initialisation 
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring with SHAP explanations",
    version="1.0.0"
)


# Load artefacts once on startup 
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

ensemble_model = joblib.load(f'{MODEL_DIR}/ensemble_model.joblib')
scaler         = joblib.load(f'{MODEL_DIR}/scaler.joblib')
threshold      = joblib.load(f'{MODEL_DIR}/threshold.joblib')

xgb_model      = ensemble_model.named_estimators_['xgb']
explainer      = shap.TreeExplainer(xgb_model)
feature_names  = get_feature_names()

print("All artefacts loaded. API ready.")


# Request schema 
class TransactionRequest(BaseModel):
    """
    Raw transaction fields as received from the payment system.
    Time and Amount are the original unengineered values —
    feature engineering happens inside the API.
    """
    Time:   float = Field(..., description="Seconds elapsed since first transaction")
    Amount: float = Field(..., description="Transaction amount in euros")
    V1:     float
    V2:     float
    V3:     float
    V4:     float
    V5:     float
    V6:     float
    V7:     float
    V8:     float
    V9:     float
    V10:    float
    V11:    float
    V12:    float
    V13:    float
    V14:    float
    V15:    float
    V16:    float
    V17:    float
    V18:    float
    V19:    float
    V20:    float
    V21:    float
    V22:    float
    V23:    float
    V24:    float
    V25:    float
    V26:    float
    V27:    float
    V28:    float


# Response schema 
class TransactionResponse(BaseModel):
    """
    Fraud scoring result with confidence tier and SHAP explanation.
    """
    transaction_id:   Optional[str]
    decision:         str
    fraud_probability: float
    confidence_tier:  str
    threshold_used:   float
    top_features:     list


# Confidence tier logic 
def get_confidence_tier(prob: float, decision: str) -> str:
    """
    Translate raw fraud probability into a human-readable
    confidence tier for downstream fraud analyst triage.

    Fraud tiers:
        HIGH   — model is very confident this is fraud (>= 0.90)
        MEDIUM — model suspects fraud but has some uncertainty (0.77–0.90)

    Legitimate tiers:
        CLEAR      — model is very confident this is legitimate (<= 0.30)
        REVIEW     — low fraud probability but worth monitoring (0.30–0.77)
    """
    if decision == 'FRAUD':
        return 'HIGH'   if prob >= 0.90 else 'MEDIUM'
    else:
        return 'CLEAR'  if prob <= 0.30 else 'REVIEW'


# Health check endpoint 
@app.get('/health')
def health():
    return {
        'status':    'healthy',
        'model':     type(ensemble_model).__name__,
        'threshold': threshold
    }


# Prediction endpoint 
@app.post('/predict', response_model=TransactionResponse)
def predict(request: TransactionRequest):
    """
    Score a single transaction for fraud.

    Pipeline:
        1. Convert request to dataframe
        2. Apply feature engineering (Time → hour_sin/cos, Amount → log)
        3. Scale using saved RobustScaler
        4. Get fraud probability from ensemble
        5. Apply threshold to make binary decision
        6. Compute SHAP explanation from XGBoost base model
        7. Return structured response
    """
    try:
        # Step 1 — Raw request to dataframe 
        raw = pd.DataFrame([request.dict()])

        # Step 2 — Feature engineering 
        engineered = engineer_features(raw)

        # Step 3 — Scale 
        scaled = scale_single(scaler, engineered.iloc[0].to_dict(), feature_names)

        # Step 4 — Predict
        fraud_prob = float(
            ensemble_model.predict_proba(scaled)[0, 1]
        )

        # Step 5 — Apply threshold
        prediction = apply_threshold(
            np.array([fraud_prob]), threshold
        )[0]
        decision   = 'FRAUD' if prediction == 1 else 'LEGITIMATE'

        # Step 6 — SHAP explanation 
        shap_values  = explainer.shap_values(scaled)
        top_features = format_shap_explanation(
            shap_values[0], feature_names, top_n=10
        )

        # Step 7 — Build response 
        return TransactionResponse(
            transaction_id    = None,
            decision          = decision,
            fraud_probability = round(fraud_prob, 4),
            confidence_tier   = get_confidence_tier(fraud_prob, decision),
            threshold_used    = threshold,
            top_features      = top_features
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))