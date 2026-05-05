# src/train.py
# Model training and serialisation 
import pandas as pd
import numpy as np
import joblib
import os

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

from src.features import engineer_features
from src.preprocess import split_data, fit_scaler, apply_scaler, apply_smote


def build_model() -> StackingClassifier:
    """
    Construct the stacking ensemble architecture.
    Base models: XGBoost + Random Forest
    Meta-learner: Logistic Regression

    Returns:
        Unfitted StackingClassifier
    """
    base_models = [
        ('rf', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )),
        ('xgb', XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )),
    ]

    meta_learner = LogisticRegression()

    return StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )


def train(data_path: str, models_dir: str = 'models') -> dict:
    """
    Full training pipeline — loads raw data, engineers features,
    preprocesses, trains the ensemble, and serialises all artefacts.

    Args:
        data_path:  Path to raw creditcard.csv
        models_dir: Directory to write serialised artefacts

    Returns:
        Dict containing trained model, scaler, and threshold
    """
    # Load and clean 
    print("Loading data...")
    df = pd.read_csv(data_path)
    df = df.drop_duplicates()
    print(f"  Rows after deduplication: {len(df):,}")

    # Feature engineering 
    print("Engineering features...")
    df = engineer_features(df)

    # Split
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    # Scale 
    print("Scaling features...")
    scaler    = fit_scaler(X_train)
    X_train_s = apply_scaler(scaler, X_train)
    X_test_s  = apply_scaler(scaler, X_test)

    # SMOTE 
    print("Applying SMOTE...")
    X_res, y_res = apply_smote(X_train_s, y_train)
    print(f"  Training shape after SMOTE: {X_res.shape}")

    # Train 
    print("Training stacking ensemble (this will take ~30 minutes)...")
    model = build_model()
    model.fit(X_res, y_res)
    print("  Training complete.")

    # Serialise 
    print("Serialising artefacts...")
    os.makedirs(models_dir, exist_ok=True)
    threshold = 0.77

    joblib.dump(model,     f'{models_dir}/ensemble_model.joblib')
    joblib.dump(scaler,    f'{models_dir}/scaler.joblib')
    joblib.dump(threshold, f'{models_dir}/threshold.joblib')
    print(f"  Artefacts saved to {models_dir}/")

    return {
        'model':     model,
        'scaler':    scaler,
        'threshold': threshold
    }


if __name__ == '__main__':
    train(data_path='data/creditcard.csv')