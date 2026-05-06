# tests/test_predictor.py
# ── Integration tests for FraudPredictor ──────────────────────────
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.predictor import FraudPredictor


# Fixtures and test classes for FraudPredictor integration tests  --- IGNORE ---
@pytest.fixture(scope='module')
def predictor():
    """Load predictor once for all tests in this module."""
    return FraudPredictor()


@pytest.fixture
def legitimate_transaction():
    """Low-risk transaction with neutral feature values."""
    data = {f'V{i}': 0.0 for i in range(1, 29)}
    data.update({'Time': 50000.0, 'Amount': 10.0})
    return data


@pytest.fixture
def suspicious_transaction():
    """High-risk transaction matching known fraud patterns."""
    data = {f'V{i}': 0.0 for i in range(1, 29)}
    data.update({
        'Time':   406.0,
        'Amount': 229.15,
        'V1':  -3.043541, 'V2':  -3.157307, 'V3':   1.088463,
        'V4':   2.288282, 'V5':   1.359805, 'V6':  -1.064823,
        'V7':  -3.216816, 'V8':   0.963958, 'V9':  -4.498295,
        'V10': -1.903324, 'V11':  1.453888, 'V12': -2.833819,
        'V13': -0.764650, 'V14': -4.941888, 'V15':  0.392831,
        'V16': -1.140788, 'V17': -2.459499, 'V18': -1.637940,
        'V19':  0.774543, 'V20':  0.034249, 'V21':  0.641673,
        'V22':  0.339485, 'V23': -0.182899, 'V24':  0.551707,
        'V25':  0.239334, 'V26':  0.406271, 'V27':  0.222938,
        'V28':  0.027510
    })
    return data


# Tests for predictor loading and artefact integrity
class TestFraudPredictorLoading:

    def test_model_loads(self, predictor):
        assert predictor.model is not None

    def test_threshold_correct(self, predictor):
        assert predictor.threshold == 0.77

    def test_feature_names_count(self, predictor):
        assert len(predictor.feature_names) == 31


class TestPredictOutput:

    def test_response_has_required_keys(self, predictor, legitimate_transaction):
        result = predictor.predict(legitimate_transaction)
        required = {
            'decision', 'fraud_probability',
            'confidence_tier', 'threshold_used', 'top_features'
        }
        assert required.issubset(result.keys())

    def test_decision_is_valid_string(self, predictor, legitimate_transaction):
        result = predictor.predict(legitimate_transaction)
        assert result['decision'] in ('FRAUD', 'LEGITIMATE')

    def test_fraud_probability_in_range(self, predictor, legitimate_transaction):
        result = predictor.predict(legitimate_transaction)
        assert 0.0 <= result['fraud_probability'] <= 1.0

    def test_confidence_tier_valid(self, predictor, legitimate_transaction):
        result = predictor.predict(legitimate_transaction)
        assert result['confidence_tier'] in ('HIGH', 'MEDIUM', 'REVIEW', 'CLEAR')

    def test_top_features_count(self, predictor, legitimate_transaction):
        result = predictor.predict(legitimate_transaction)
        assert len(result['top_features']) == 10

    def test_top_features_have_correct_keys(self, predictor, legitimate_transaction):
        result   = predictor.predict(legitimate_transaction)
        feature  = result['top_features'][0]
        assert 'feature'    in feature
        assert 'shap_value' in feature
        assert 'direction'  in feature

    def test_direction_values_valid(self, predictor, legitimate_transaction):
        result = predictor.predict(legitimate_transaction)
        valid  = {'toward_fraud', 'toward_legitimate'}
        for f in result['top_features']:
            assert f['direction'] in valid

    def test_threshold_used_matches_loaded(self, predictor, legitimate_transaction):
        result = predictor.predict(legitimate_transaction)
        assert result['threshold_used'] == predictor.threshold

    def test_legitimate_transaction_clears(self, predictor, legitimate_transaction):
        result = predictor.predict(legitimate_transaction)
        assert result['fraud_probability'] < 0.77

    def test_suspicious_transaction_scores_high(self, predictor, suspicious_transaction):
        result = predictor.predict(suspicious_transaction)
        assert result['fraud_probability'] > 0.40