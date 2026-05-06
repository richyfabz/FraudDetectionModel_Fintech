# api/schema.py
# ── Request and Response schemas ───────────────────────────────────
from pydantic import BaseModel, Field
from typing import Optional


class TransactionRequest(BaseModel):
    """
    Raw transaction fields as received from the payment system.
    Time and Amount are original unengineered values.
    Feature engineering happens inside the API pipeline.
    """
    Time:   float = Field(..., description="Seconds elapsed since first transaction in dataset")
    Amount: float = Field(..., ge=0, description="Transaction amount in euros, must be non-negative")
    V1:  float; V2:  float; V3:  float; V4:  float
    V5:  float; V6:  float; V7:  float; V8:  float
    V9:  float; V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 406.0,
                "Amount": 229.15,
                "V1": -3.043541, "V2": -3.157307, "V3": 1.088463,
                "V4": 2.288282,  "V5": 1.359805,  "V6": -1.064823,
                "V7": -3.216816, "V8": 0.963958,  "V9": -4.498295,
                "V10": -1.903324,"V11": 1.453888,  "V12": -2.833819,
                "V13": -0.764650,"V14": -4.941888, "V15": 0.392831,
                "V16": -1.140788,"V17": -2.459499, "V18": -1.637940,
                "V19": 0.774543, "V20": 0.034249,  "V21": 0.641673,
                "V22": 0.339485, "V23": -0.182899, "V24": 0.551707,
                "V25": 0.239334, "V26": 0.406271,  "V27": 0.222938,
                "V28": 0.027510
            }
        }


class FeatureContribution(BaseModel):
    """Single feature's contribution to a fraud prediction."""
    feature:    str
    shap_value: float
    direction:  str


class TransactionResponse(BaseModel):
    """
    Complete fraud scoring result returned by /predict.
    """
    transaction_id:    Optional[str]
    decision:          str
    fraud_probability: float
    confidence_tier:   str
    threshold_used:    float
    top_features:      list[FeatureContribution]