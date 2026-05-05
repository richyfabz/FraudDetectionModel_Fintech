# src/features.py
# ── Feature engineering for fraud detection pipeline ──────────────
import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a raw dataframe.

    Steps:
        1. Convert raw Time (seconds) to hour of day
        2. Cyclical encoding of hour using sin/cos
        3. Log transformation of Amount
        4. Drop original Time, Amount, and intermediate hour columns

    Args:
        df: Raw dataframe with Time and Amount columns present

    Returns:
        Dataframe with engineered features and originals removed
    """
    df = df.copy()

    # ── Time → cyclical hour encoding ─────────────────────────────
    df['hour']     = (df['Time'] / 3600) % 24
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # ── Amount → log transform ────────────────────────────────────
    df['amount_log'] = np.log1p(df['Amount'])

    # ── Drop raw and intermediate columns ─────────────────────────
    df = df.drop(['Time', 'Amount', 'hour'], axis=1, errors='ignore')

    return df


def get_feature_names() -> list:
    """
    Return the expected feature column names after engineering.
    Used by the API to validate incoming request fields.
    """
    v_features   = [f'V{i}' for i in range(1, 29)]
    eng_features = ['hour_sin', 'hour_cos', 'amount_log']
    return v_features + eng_features