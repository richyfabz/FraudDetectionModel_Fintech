# src/preprocess.py
#  Preprocessing pipeline for fraud detection 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE


def split_data(
    df: pd.DataFrame,
    target_col: str = 'Class',
    test_size: float = 0.2,
    random_state: int = 42
) -> tuple:
    """
    Separate features from target and perform stratified train/test split.

    Args:
        df:           Engineered dataframe (output of engineer_features)
        target_col:   Name of the target column
        test_size:    Proportion of data reserved for testing
        random_state: Reproducibility seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def fit_scaler(X_train: pd.DataFrame) -> RobustScaler:
    """
    Fit a RobustScaler on training data only.

    Args:
        X_train: Unscaled training features

    Returns:
        Fitted RobustScaler instance
    """
    scaler = RobustScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(
    scaler: RobustScaler,
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply a pre-fitted scaler to any feature matrix.
    Used for both training and inference — never fits again.

    Args:
        scaler: Already fitted RobustScaler
        X:      Feature matrix to scale

    Returns:
        Scaled dataframe with original column names preserved
    """
    scaled = scaler.transform(X)
    return pd.DataFrame(scaled, columns=X.columns)


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 1.0,
    random_state: int = 42
) -> tuple:
    """
    Apply SMOTE to training data only to balance the fraud class.
    Never called during inference.

    Args:
        X_train:           Scaled training features
        y_train:           Training labels
        sampling_strategy: Target fraud/legitimate ratio
        random_state:      Reproducibility seed

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res


def scale_single(
    scaler: RobustScaler,
    transaction: dict,
    feature_names: list
) -> pd.DataFrame:
    """
    Scale a single incoming transaction for API inference.
    Converts raw dict to a one-row dataframe and applies the fitted scaler.

    Args:
        scaler:        Fitted RobustScaler loaded from models/
        transaction:   Dict of feature_name → value from API request
        feature_names: Ordered list of expected feature names

    Returns:
        One-row scaled dataframe ready for model.predict_proba()
    """
    row = pd.DataFrame([transaction])[feature_names]
    return apply_scaler(scaler, row)