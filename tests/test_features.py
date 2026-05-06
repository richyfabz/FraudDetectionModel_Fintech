# tests/test_features.py
# ── Unit tests for feature engineering ────────────────────────────
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import engineer_features, get_feature_names


# Fixtures for raw transactions with known values to test transformations against
@pytest.fixture
def raw_transaction():
    """Single raw transaction row matching the original dataset schema."""
    data = {f'V{i}': 0.0 for i in range(1, 29)}
    data.update({'Time': 3600.0, 'Amount': 100.0, 'Class': 0})
    return pd.DataFrame([data])


@pytest.fixture
def raw_transaction_midnight():
    """Transaction at midnight — Time=0 seconds."""
    data = {f'V{i}': 0.0 for i in range(1, 29)}
    data.update({'Time': 0.0, 'Amount': 0.0, 'Class': 0})
    return pd.DataFrame([data])


# Tests for engineer_features 
class TestEngineerFeatures:

    def test_output_columns_correct(self, raw_transaction):
        result = engineer_features(raw_transaction)
        expected = get_feature_names() + ['Class']
        assert list(result.columns) == expected

    def test_time_and_amount_dropped(self, raw_transaction):
        result = engineer_features(raw_transaction)
        assert 'Time'   not in result.columns
        assert 'Amount' not in result.columns
        assert 'hour'   not in result.columns

    def test_hour_sin_cos_at_midnight(self, raw_transaction_midnight):
        result = engineer_features(raw_transaction_midnight)
        assert round(result['hour_sin'].iloc[0], 6) == 0.0
        assert round(result['hour_cos'].iloc[0], 6) == 1.0

    def test_hour_sin_cos_at_noon(self, raw_transaction):
        # Time=3600 → hour=1, not noon. Use Time=43200 for noon (12h)
        data = {f'V{i}': 0.0 for i in range(1, 29)}
        data.update({'Time': 43200.0, 'Amount': 100.0, 'Class': 0})
        df     = pd.DataFrame([data])
        result = engineer_features(df)
        # hour=12: sin(2π*12/24) = sin(π) ≈ 0, cos(π) = -1
        assert abs(result['hour_sin'].iloc[0]) < 1e-6
        assert round(result['hour_cos'].iloc[0], 6) == -1.0

    def test_amount_log_transform(self, raw_transaction):
        result = engineer_features(raw_transaction)
        expected = np.log1p(100.0)
        assert round(result['amount_log'].iloc[0], 6) == round(expected, 6)

    def test_amount_zero_no_error(self, raw_transaction_midnight):
        # log1p(0) = 0 — must not raise or return -inf
        result = engineer_features(raw_transaction_midnight)
        assert result['amount_log'].iloc[0] == 0.0

    def test_does_not_mutate_input(self, raw_transaction):
        original_cols = list(raw_transaction.columns)
        engineer_features(raw_transaction)
        assert list(raw_transaction.columns) == original_cols


class TestGetFeatureNames:

    def test_returns_31_features(self):
        assert len(get_feature_names()) == 31

    def test_v_features_present(self):
        names = get_feature_names()
        for i in range(1, 29):
            assert f'V{i}' in names

    def test_engineered_features_present(self):
        names = get_feature_names()
        assert 'hour_sin'   in names
        assert 'hour_cos'   in names
        assert 'amount_log' in names

    def test_order_v_features_first(self):
        names = get_feature_names()
        assert names[:28] == [f'V{i}' for i in range(1, 29)]