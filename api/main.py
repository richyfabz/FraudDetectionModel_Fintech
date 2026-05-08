# api/main.py
# Fraud Detection REST API 
from fastapi import FastAPI, HTTPException
from api.schema import TransactionRequest, TransactionResponse
from api.predictor import FraudPredictor

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring with SHAP explanations",
    version="1.0.0"
)

predictor = FraudPredictor()


@app.get('/health')
def health():
    return {
        'status':    'healthy',
        'model':     type(predictor.model).__name__,
        'threshold': predictor.threshold
    }


@app.post('/predict', response_model=TransactionResponse)
def predict(request: TransactionRequest):
    try:
        return predictor.predict(request.dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))