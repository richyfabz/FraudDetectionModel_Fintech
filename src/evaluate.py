# src/evaluate.py
# Model evaluation and threshold tuning 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> dict:
    """
    Compute all evaluation metrics for a trained model.

    Args:
        y_true: True labels
        y_pred: Binary predictions at current threshold
        y_prob: Fraud probabilities from predict_proba

    Returns:
        Dict of metric name → value
    """
    return {
        'auc_roc':  roc_auc_score(y_true, y_prob),
        'f1':       f1_score(y_true, y_pred),
        'report':   classification_report(
                        y_true, y_pred,
                        target_names=['Legitimate', 'Fraud']
                    ),
        'confusion': confusion_matrix(y_true, y_pred)
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    step: float = 0.01
) -> tuple:
    """
    Sweep classification thresholds and return the one
    that maximises F1-Score on the provided labels.

    Args:
        y_true: True labels
        y_prob: Fraud probabilities from predict_proba
        step:   Threshold increment for sweep

    Returns:
        Tuple of (best_threshold, best_f1)
    """
    thresholds = np.arange(0.01, 1.0, step)
    f1_scores  = [
        f1_score(y_true, (y_prob >= t).astype(int))
        for t in thresholds
    ]
    best_idx   = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def apply_threshold(
    y_prob: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Convert fraud probabilities to binary predictions
    using a specified threshold.
    Called by the API on every incoming transaction.

    Args:
        y_prob:    Fraud probability scores from predict_proba
        threshold: Decision boundary — scores above this = fraud

    Returns:
        Binary prediction array (0 = legitimate, 1 = fraud)
    """
    return (y_prob >= threshold).astype(int)


def print_results(metrics: dict) -> None:
    """
    Print evaluation results to stdout in a readable format.

    Args:
        metrics: Output of compute_metrics()
    """
    print("=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"AUC-ROC:  {metrics['auc_roc']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    print()
    print(metrics['report'])


def plot_confusion_matrix(
    metrics: dict,
    save_path: str = None
) -> None:
    """
    Render and optionally save the confusion matrix plot.

    Args:
        metrics:   Output of compute_metrics()
        save_path: If provided, saves the plot to this path
    """
    disp = ConfusionMatrixDisplay(
        confusion_matrix=metrics['confusion'],
        display_labels=['Legitimate', 'Fraud']
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def format_shap_explanation(
    shap_values: np.ndarray,
    feature_names: list,
    top_n: int = 10
) -> list:
    """
    Format SHAP values for a single transaction into a
    ranked list of feature contributions.
    Called by the API to embed explanations in every response.

    Args:
        shap_values:   1D array of SHAP values for one transaction
        feature_names: Ordered list of feature names
        top_n:         Number of top contributors to return

    Returns:
        List of dicts sorted by absolute SHAP value descending:
        [{'feature': 'V14', 'shap_value': 5.15, 'direction': 'toward_fraud'}, ...]
    """
    contributions = [
        {
            'feature':   name,
            'shap_value': round(float(val), 4),
            'direction':  'toward_fraud' if val > 0 else 'toward_legitimate'
        }
        for name, val in zip(feature_names, shap_values)
    ]

    contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
    return contributions[:top_n]