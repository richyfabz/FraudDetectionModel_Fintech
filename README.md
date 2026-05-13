# Fraud Detection Model Fintech MLOps Stack

A production-grade credit card fraud detection system combining
unsupervised anomaly detection with automated ensemble learning.
Built on 284,807 real European cardholder transactions with a
full-stack deployment Flask REST API + React frontend.

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
- [Docker Deployment](#docker-deployment)

---

## Project Overview

Credit card fraud detection is a class imbalance problem at an industrial scale 99.83% of transactions are legitimate and only 0.17% are fraudulent. A naive model that predicts every transaction as legitimate achieves 99.83% accuracy while catching zero fraud. This project solves that problem through a combination of:

- **Stacking ensemble** to combine the complementary strengths of unsupervised learming, gradient boosting and bagging
- **Threshold tuning** to find the optimal decision boundary that maximises F1-Score
- **FlaskAPI** to serve all of the above as a production REST endpoint with structured JSON responses

---

## Architecture
```
Raw Transaction (30 features)
│
▼
StandardScaler — normalise all features to equal magnitude
│
▼
Isolation Forest — unsupervised anomaly score
trained on 226,602 legitimate transactions only
│
▼
AutoGluon WeightedEnsemble — supervised classification
31 features (30 original + anomaly score)
RandomForest + LightGBM + CatBoost + ExtraTrees
│
▼
Threshold 0.397 — tuned to maximise F1-Score
│
▼
FRAUD / LEGITIMATE + confidence tier + anomaly score
```
## Dataset

| Property | Value |
|---|---|
| Source | Kaggle Credit Card Fraud Detection |
| Total transactions | 284,807 |
| After deduplication | 283,726 |
| Features | V1–V28 (PCA), Time, Amount |
| Target | Class (0=Legitimate, 1=Fraud) |
| Fraud rate | 0.17% (473 cases) |
| Train split |  198,608 rows |
| Test split | 585,118 rows |

The V1–V28 features are PCA-transformed components from the original transaction data, provided pre-anonymised by the dataset authors. Time and Amount are the only original unmasked features.

## Tech Stack

### Machine Learning
```
| Component | Technology |
|---|---|
| Anomaly Detection | Isolation Forest (sklearn) |
| AutoML Framework | AutoGluon 1.5.0 |
| Base Models | Random Forest, LightGBM, CatBoost, ExtraTrees, KNN |
| Meta-Learner | WeightedEnsemble L2 |
| Feature Scaling | StandardScaler |
| Serialisation | Joblib |
```
### Backend
```
| Component | Technology |
|---|---|
| API Framework | Flask |
| Cross-Origin | Flask-CORS |
| Data Processing | Pandas, NumPy |
| Python Version | 3.10+ |
```

### Frontend
```
| Component | Technology |
|---|---|
| Framework | React 19 |
| Routing | React Router v6 |
| Animations | Framer Motion |
| Styling | Tailwind CSS v3 |
| HTTP Client | Axios |
| Icons | Lucide React |
```

## Project Structure
```
FraudDetection/
│
├── notebooks/
│   ├── 01_data_ingestion_and_eda.ipynb
│   ├── 02_preprocessing_and_feature_engineering.ipynb
│   ├── 03_model_training_and_evaluation.ipynb
│   └── 04_explainability_and_shap.ipynb
│
├── flask_app/
│   ├── app.py              # Flask API — 5 endpoints
│   └── predictor.py        # FraudPredictor class — full pipeline
│
├── frontend/
│   ├── src/
│   │   ├── App.js          # Router setup
│   │   ├── components/
│   │   │   └── Navbar.jsx
│   │   ├── pages/
│   │   │   ├── Landing.jsx
│   │   │   ├── HowItWorks.jsx
│   │   │   ├── FeatureGuide.jsx
│   │   │   ├── Analyser.jsx
│   │   │   └── BulkUpload.jsx
│   │   └── sampleTransactions.json
│   └── package.json
│
├── models/
│   ├── autogluon/          # AutoGluon model directory
│   ├── isolation_forest.joblib
│   ├── scaler.joblib
│   ├── threshold.joblib
│   └── anomaly_norm_params.joblib
│
├── data/
│   └── creditcard.csv      # Not committed — download from Kaggle
│
├── src/
│   ├── features.py
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── tests/
│   ├── test_features.py
│   └── test_predictor.py
│
├── Dockerfile
├── requirements.txt
└── README.md
```

## Pipeline Stages

### Stage 1 — Data Ingestion and EDA
- 284,807 transactions loaded from Kaggle Credit Card Fraud dataset
- 1,081 duplicates removed 283,726 clean rows
- Zero null values confirmed across all 31 columns
- Class imbalance: 99.83% legitimate vs 0.17% fraud 578:1 ratio
- Outlier analysis: fraud rate in high-value transactions 65% above average
- Key predictors identified: V14, V17, V4, V12

### Stage 2 — Preprocessing
- All 30 features kept Time, Amount, V1–V28 — nothing dropped
- Stratified 80/20 train/test split preserving 0.1667% fraud rate
- StandardScaler fitted on training data only test set never touches scaler fitting
- 226,602 legitimate transactions isolated for Isolation Forest training

### Stage 3 — Isolation Forest
- 100 isolation trees, contamination=0.001667 (known fraud rate)
- Trained exclusively on legitimate transactions fully unsupervised
- Anomaly score computed for all transactions and normalised to [0,1]
- Feature importance extracted Time ranked 2nd, Amount 11th

### Stage 4 — AutoGluon Ensemble
- 31 features: 30 original + IF anomaly score
- 10-minute training budget, best_quality preset
- Models trained: Random Forest, LightGBM, CatBoost, ExtraTrees, KNN
- 8-fold bagging per model
- WeightedEnsemble L2 selected as best model

### Stage 5 — Threshold Tuning
- 200 candidate thresholds swept from 0 to 1
- F1-Score maximised at threshold 0.397
- Default 0.5 threshold gives F1 0.8855
- Tuned threshold gives F1 0.8939

### Stage 6 — Flask REST API
- 5 endpoints: /api/health, /api/models, /api/features,
  /api/predict, /api/predict/batch
- Model selection user can choose any AutoGluon model
- Batch scoring accepts any CSV format, fills missing columns with 0
- Safe feature mapping missing features default to 0

### Stage 7 — React Frontend
- 5 pages: Landing, How It Works, Feature Guide, Analyser, Bulk Upload
- Live model selector fetches available models from Flask
- 20 real test transactions embedded for demo (10 fraud + 10 legitimate)
- Batch CSV upload with results table, summary stats, and CSV download
- Framer Motion animations throughout

---

## Model Performance
```
| Metric | Value |
|---|---|
| AUC-ROC | 0.9610 |
| F1-Score (tuned threshold) | 0.8939 |
| Fraud Precision | 97% |
| Fraud Recall | 82% |
| Fraud Cases Caught | 116 / 142 |
| False Alarms | 4 / 84,976 |
| Training Accuracy | 99.9995% |
| Test Accuracy | 99.9648% |
| Train/Test Gap | 0.000347 — no overfitting |
| Test Transactions | 85,118 |
```

## API Reference

### GET /api/health
```json
{
  "status": "active",
  "model_loaded": true,
  "threshold": 0.397
}
```
### GET /api/models
Returns all AutoGluon models available for selection.

### GET /api/features
Returns metadata for all 30 features including descriptions
and importance ratings.

### POST /api/predict
```json
// Request
{
  "Time": 406.0,
  "Amount": 229.15,
  "V1": -3.043541,
  "...": "V2 through V28",
  "selected_model": "WeightedEnsemble_L2"
}

// Response
{
  "decision": "FRAUD",
  "fraud_probability": 61.61,
  "confidence_tier": "MEDIUM",
  "threshold_used": 39.7,
  "anomaly_score": 41.8,
  "model_used": "WeightedEnsemble_L2",
  "raw_input": {}
}
```

### POST /api/predict/batch
Accepts multipart CSV upload or JSON array.
Returns array of predictions with summary statistics.


## Confidence Tiers
```
| Tier | Condition | Action |
|---|---|---|
| HIGH | FRAUD and P ≥ 90% | Immediate block |
| MEDIUM | FRAUD and P < 90% | Investigate |
| REVIEW | LEGITIMATE and P > 30% | Monitor |
| CLEAR | LEGITIMATE and P ≤ 30% | No action |
```



## Running the Project

**1. Clone and set up environment**
```bash
git clone https://github.com/richyfabz/FraudDetectionModel_Fintech.git
cd FraudDetection
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Download the dataset**

Download `creditcard.csv` from
[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place it at `data/creditcard.csv`.

**3. Train the model (optional — models already serialised)**
```bash
# Run notebooks 01 through 03 in order
# Or retrain from command line:
python -m src.train
```

**4. Start Flask API**
```bash
cd flask_app
source ../venv/bin/activate
python app.py
# Running on http://127.0.0.1:5000
```

**5. Start React frontend**
```bash
cd frontend
npm install
npm start
# Running on http://localhost:3000
```

---

## Key Design Decisions

**Isolation Forest on legitimate transactions only** —
Training on all transactions would teach the model that fraud
is "normal." Training on legitimate transactions only means
any deviation from normal behaviour — including new fraud
patterns never seen before — gets flagged.

**Anomaly score as a feature** — Rather than using Isolation
Forest as the final classifier (which gives poor precision),
its output is passed to AutoGluon as a pre-computed signal.
AutoGluon learns when to trust and when to discount it.

**Time and Amount kept raw** — Previous architecture dropped
both. Isolation Forest ranked Time 2nd most important.
Dropping them caused the model to miss 4 out of 10 fraud
cases in the demonstration test.

**No SMOTE** — The Isolation Forest anomaly score already
encodes the imbalance signal. AutoGluon's scale_pos_weight
handles class weighting internally. SMOTE generated synthetic
overlap that caused data leakage.

**Threshold tuning to 0.397** — The default 0.5 threshold
assumes balanced classes. Sweeping 200 candidate thresholds
and selecting on F1-Score finds the boundary optimal for a
578:1 imbalanced distribution.


## Dataset

- Source: Kaggle Credit Card Fraud Detection
- Transactions: 284,807 (283,726 after deduplication)
- Features: V1–V28 (PCA), Time, Amount
- Target: Class (0=Legitimate, 1=Fraud)
- Fraud rate: 0.1667% (473 cases)
- Time period: 48 hours of European cardholder transactions
