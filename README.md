# Fraud Detection Model Fintech MLOps Stack

A production-grade fraud detection system built on the Kaggle Credit Card Fraud Detection dataset. The system combines a stacking ensemble of XGBoost and Random Forest models with real-time SHAP explanations, served via a FastAPI REST API. Every fraud decision includes a full audit trail of feature contributions meeting the explainability requirements of production fintech compliance.


## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
- [Model Performance](#model-performance)
- [API Reference](#api-reference)
- [Running the Project](#running-the-project)
- [Running Tests](#running-tests)
- [Docker Deployment](#docker-deployment)

---

## Project Overview

Credit card fraud detection is a class imbalance problem at industrial scale — 99.83% of transactions are legitimate and only 0.17% are fraudulent. A naive model that predicts every transaction as legitimate achieves 99.83% accuracy while catching zero fraud. This project solves that problem through a combination of:

- **SMOTE** to synthetically balance the training set without touching the test set
- **Stacking ensemble** to combine the complementary strengths of gradient boosting and bagging
- **Threshold tuning** to find the optimal decision boundary that maximises F1-Score
- **SHAP explainability** to produce a per-prediction audit trail for every fraud decision
- **FastAPI** to serve all of the above as a production REST endpoint with structured JSON responses

---

## Architecture

TRAINING PIPELINE (offline, runs once)
──────────────────────────────────────────────────────────────────
creditcard.csv
│
▼
src/features.py     engineer_features()
│                   Time → hour_sin/cos (cyclical encoding)
│                   Amount → amount_log (log1p transform)
│                   Drops: Time, Amount, hour
▼
src/preprocess.py   split_data()
│                   Stratified 80/20 train/test split
│                   random_state=42, stratify=y
▼
src/preprocess.py   fit_scaler() → apply_scaler()
│                   RobustScaler fitted on training data only
│                   Robust to high-value outliers retained as fraud signal
▼
src/preprocess.py   apply_smote()
│                   SMOTE sampling_strategy=1.0
│                   226,602 legitimate → balanced with 226,602 synthetic fraud
│                   Training set: 453,204 rows
▼
src/train.py        build_model() → train()
│                   Level 0: XGBoost + Random Forest (cv=5)
│                   Level 1: Logistic Regression meta-learner
│                   StackingClassifier, n_jobs=-1
▼
models/
├── ensemble_model.joblib   (~15.7 MB)
├── scaler.joblib           (~1.6 KB)
└── threshold.joblib        (~0.1 KB)
INFERENCE PIPELINE (live, runs on every API request)
──────────────────────────────────────────────────────────────────
POST /predict  ←  raw transaction JSON
│
▼
api/schema.py           TransactionRequest validation (Pydantic)
│                   Validates all 30 fields present and typed correctly
▼
src/features.py         engineer_features()
│                   Same function as training — guaranteed consistency
▼
src/preprocess.py       scale_single()
│                   Applies saved RobustScaler — never refits
▼
models/ensemble_model   predict_proba()
│                   Returns P(fraud) between 0.0 and 1.0
▼
src/evaluate.py         apply_threshold()
│                   Threshold = 0.77 (tuned on test set)
│                   P(fraud) >= 0.77 → FRAUD
▼
src/evaluate.py         format_shap_explanation()
│                   TreeExplainer on XGBoost base model
│                   Top 10 feature contributions ranked by |SHAP value|
▼
api/schema.py           TransactionResponse
decision + probability + confidence_tier + explanation

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle Credit Card Fraud Detection |
| Total transactions | 284,807 |
| After deduplication | 283,726 |
| Features | V1–V28 (PCA), Time, Amount |
| Target | Class (0=Legitimate, 1=Fraud) |
| Fraud rate | 0.17% (473 cases) |
| Train split | 226,980 rows |
| Test split | 56,746 rows |
| After SMOTE | 453,204 training rows (balanced) |

The V1–V28 features are PCA-transformed components from the original transaction data, provided pre-anonymised by the dataset authors. Time and Amount are the only original unmasked features.

## Tech Stack

| Layer | Library | Version |
|---|---|---|
| Language | Python | 3.10+ |
| Data processing | Pandas, NumPy | — |
| Preprocessing | Scikit-learn | — |
| Imbalance handling | Imbalanced-learn | — |
| Base model 1 | XGBoost | — |
| Base model 2 | Scikit-learn RandomForest | — |
| Meta-learner | Scikit-learn LogisticRegression | — |
| Explainability | SHAP | — |
| API framework | FastAPI | — |
| API server | Uvicorn | — |
| Validation | Pydantic | — |
| Serialisation | Joblib | — |
| Testing | Pytest | — |
| Containerisation | Docker | — |


## Project Structure

FraudDetection/
│
├── api/
│   ├── init.py
│   ├── main.py             # FastAPI app — routing only
│   ├── predictor.py        # FraudPredictor class — all inference logic
│   └── schema.py           # Pydantic request and response schemas
│
├── data/
│   └── creditcard.csv      # Raw dataset (not committed to git)
│
├── models/
│   ├── ensemble_model.joblib
│   ├── scaler.joblib
│   └── threshold.joblib
│
├── notebooks/
│   ├── 01_data_ingestion_and_eda.ipynb
│   ├── 02_preprocessing_and_feature_engineering.ipynb
│   ├── 03_model_training_and_evaluation.ipynb
│   └── 04_explainability_and_shap.ipynb
│
├── Plots/
│   ├── evaluation_threshold_tuning.png
│   ├── shap_global_importance.png
│   ├── shap_summary_dot.png
│   └── shap_waterfall_fraud.png
│
├── src/
│   ├── init.py
│   ├── features.py         # Feature engineering
│   ├── preprocess.py       # Scaling and SMOTE
│   ├── train.py            # Model training and serialisation
│   └── evaluate.py         # Metrics, threshold tuning, SHAP formatting
│
├── tests/
│   ├── test_features.py    # Unit tests — feature engineering
│   └── test_predictor.py   # Integration tests — full prediction pipeline
│
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt

---

## Pipeline Stages

### Stage 1 — Data Ingestion
Loaded 284,807 rows from the Kaggle dataset. Confirmed zero null values across all 31 columns. Removed 1,081 duplicate rows leaving 283,726 clean rows. Confirmed 473 fraud cases after deduplication.

### Stage 2 — Exploratory Data Analysis
Analysed class imbalance — 99.83% legitimate vs 0.17% fraud, ratio of 578:1. Confirmed Amount is right-skewed with fraud transactions having a higher mean (€122 vs €88) — log transform required. Confirmed Time distribution shows fraud is uniformly distributed across all hours while legitimate transactions peak at daytime cyclical encoding required. Detected 31,685 Amount outliers fraud rate in outlier group (0.2746%) is 65% higher than overall outliers retained as genuine fraud signal. Identified V14, V17, V12, V10 as strongest negative predictors and V11, V4 as strongest positive predictors via correlation heatmap.

### Stage 3 — Preprocessing
Stratified 80/20 train/test split preserving the 0.17% fraud rate in both splits. RobustScaler chosen over StandardScaler uses median and IQR instead of mean and standard deviation, making it robust to the high-value outliers retained as fraud signal. Scaler fitted on training data only, applied identically to test set and API inference.

### Stage 4 — Feature Engineering
Time converted to hour of day using `(Time / 3600) % 24`. Hour encoded cyclically using `sin(2π × hour/24)` and `cos(2π × hour/24)` — ensures midnight and 23:59 are numerically close. Amount log-transformed using `log1p(Amount)` compresses the right-skewed distribution and handles zero-value transactions safely. Original Time, Amount, and intermediate hour columns dropped. Final feature count: 31.

### Stage 5 — SMOTE
Applied to training set only. Generates synthetic fraud examples by interpolating between real fraud cases in feature space using k-nearest neighbours. Balances training set from 226,602 legitimate / 378 fraud to 226,602 / 226,602 — final training shape 453,204 × 31. Test set remains untouched at 56,746 rows.

### Stage 6 — Model Training
Stacking ensemble with 5-fold cross-validation. XGBoost (gradient boosting captures complex non-linear patterns) and Random Forest (bagging robust to outliers, prevents overfitting) as Level 0 base models. Logistic Regression as Level 1 meta-learner learns to weight base model predictions optimally. Training time approximately 38 minutes on 453k rows.

### Stage 7 — Evaluation and Threshold Tuning
Evaluated on 56,746 unseen test transactions. AUC-ROC 0.9755 model correctly ranks fraud above legitimate 97.55% of the time. Default threshold F1 0.8457. Threshold sweep from 0.01 to 0.99 identified 0.77 as optimal F1 improves to 0.8555. Final confusion matrix: 74 fraud caught, 21 missed, 6 false alarms, 56,645 legitimate correctly cleared.

### Stage 8 — SHAP Explainability
TreeExplainer applied to XGBoost base model on full test set. Global feature importance confirms V14 and V4 as dominant predictors. hour_cos and hour_sin rank 3rd and 4th validating the cyclical encoding decision. Per-prediction waterfall plots show individual feature contributions for audit trail generation.

### Stage 9 — FastAPI Deployment
REST API with two endpoints. All artefacts loaded once at startup. Every `/predict` response includes the fraud decision, probability, confidence tier, threshold used, and top 10 SHAP feature contributions. Full pipeline from raw JSON to structured response runs in under one second.

---

## Model Performance

| Metric | Value |
|---|---|
| AUC-ROC | 0.9755 |
| F1-Score (default threshold 0.5) | 0.8457 |
| F1-Score (tuned threshold 0.77) | 0.8555 |
| Optimal threshold | 0.77 |
| Fraud precision | 0.93 |
| Fraud recall | 0.78 |
| True positives (fraud caught) | 74 |
| False negatives (fraud missed) | 21 |
| False positives (false alarms) | 6 |
| True negatives (legitimate cleared) | 56,645 |
| Test set size | 56,746 transactions |

---

## API Reference

### GET /health

Returns API status and loaded model information.

**Response:**
```json
{
  "status": "healthy",
  "model": "StackingClassifier",
  "threshold": 0.77
}
```

### POST /predict

Score a single transaction for fraud.

**Request body:**
```json
{
  "Time": 406.0,
  "Amount": 229.15,
  "V1": -3.043541,
  "V2": -3.157307,
  "...": "V3 through V28"
}
```

**Response:**
```json
{
  "transaction_id": null,
  "decision": "FRAUD",
  "fraud_probability": 0.9998,
  "confidence_tier": "HIGH",
  "threshold_used": 0.77,
  "top_features": [
    {
      "feature": "V14",
      "shap_value": 5.15,
      "direction": "toward_fraud"
    }
  ]
}
```

**Confidence tiers:**

| Tier | Condition | Meaning |
|---|---|---|
| HIGH | FRAUD and P >= 0.90 | Immediate action required |
| MEDIUM | FRAUD and P < 0.90 | Investigate |
| REVIEW | LEGITIMATE and P > 0.30 | Monitor — borderline case |
| CLEAR | LEGITIMATE and P <= 0.30 | No action required |

**Interactive docs:** `http://127.0.0.1:8000/docs`


## Running the Project

**1. Clone the repository**
```bash
git clone https://github.com/richyfabz/FraudDetectionModel_Fintech.git
cd FraudDetection
```

**2. Create virtual environment**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add the dataset**

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/creditcard.csv`.

**5. Train the model**
```bash
python -m src.train
```

**6. Start the API**
```bash
uvicorn api.main:app --reload
```

API is live at `http://127.0.0.1:8000`

---

## Running Tests

```bash
python -m pytest tests/ -v
```

24 tests across two files unit tests for feature engineering and integration tests for the full prediction pipeline.


## Docker Deployment

Build and run the containerised API:

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

API is live at `http://localhost:8000`

---

## Key Design Decisions

**RobustScaler over StandardScaler** — high-value transactions carry genuine fraud signal and were intentionally retained. StandardScaler's mean and standard deviation would be distorted by these outliers. RobustScaler's median and IQR are unaffected.

**Cyclical time encoding** — raw Time in seconds is a weak linear feature. Sin/cos encoding of hour of day ensures midnight and 23:59 are numerically adjacent, allowing the model to detect time-of-day fraud patterns correctly. SHAP confirmed hour_cos and hour_sin rank 3rd and 4th globally.

**Threshold tuning to 0.77** — the default 0.5 threshold optimises for balanced classes. For a heavily imbalanced fraud problem, sweeping thresholds and selecting on F1-Score finds the boundary that best balances catching fraud against generating false alarms.

**SHAP on XGBoost base model** — TreeExplainer computes exact Shapley values using the tree structure directly. Applied to the XGBoost base model rather than the full stacking ensemble because TreeExplainer requires a tree-based model. XGBoost is the strongest predictor in the ensemble and its feature attributions are representative of the overall decision logic.

**FraudPredictor class** — isolates all inference logic from API routing. Independently instantiable and testable without running the HTTP server. Artefacts loaded once at startup and reused across all requests loading a 15.7MB model per request would be unacceptable latency.