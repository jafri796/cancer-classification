"""
Calibration utilities for PCam center-region detection.
Implements temperature scaling and threshold selection.

CRITICAL FOR DATA INTEGRITY:
- Temperature scaling MUST fit on VALIDATION set only (never test set)
- Thresholds MUST be optimized on VALIDATION set only
- Test set is held-out for final evaluation (no tuning)
- Any fit/optimization on test set causes data leakage
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Literal
import json
import logging
import numpy as np
import torch
import torch.nn as nn

from src.training.metrics import ClinicalValidator

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    temperature: float
    thresholds: Dict[str, float]
    target_sensitivity: float
    target_specificity: float
    fit_set: Literal["validation"] = "validation"  # CRITICAL: Document fit-set origin
    
    def to_json(self) -> str:
        return json.dumps(
            {
                "temperature": self.temperature,
                "thresholds": self.thresholds,
                "target_sensitivity": self.target_sensitivity,
                "target_specificity": self.target_specificity,
                "fit_set": self.fit_set,
                "_warning": "CRITICAL: Calibration fitted on validation set only. Never use test set for calibration.",
            },
            indent=2,
        )


class TemperatureScaler(nn.Module):
    """
    Single-parameter temperature scaling for logits.
    
    CRITICAL: Must fit on VALIDATION set only. Test set must remain held-out.
    If fit on test set, model metrics are invalid (data leakage).
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([temperature], dtype=torch.float32))
        self._is_fitted = False

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-6)

    def fit(
        self, 
        logits: np.ndarray, 
        labels: np.ndarray, 
        max_iter: int = 50,
        fit_set_name: str = "validation",
    ) -> float:
        """
        Fit temperature scaling to logits and labels.
        
        CRITICAL: Caller MUST ensure logits/labels are from VALIDATION set only.
        Never fit on test set.
        
        Args:
            logits: Model logits from validation set (N,)
            labels: Ground truth labels from validation set (N,)
            max_iter: LBFGS max iterations
            fit_set_name: Name of set used for fitting (validation/calibration)
        
        Returns:
            Fitted temperature value
        
        Raises:
            ValueError: If fit_set_name indicates test set usage (data leakage risk)
        """
        # Validate fit-set is not test
        if fit_set_name.lower() in ["test", "testset", "test_set"]:
            raise ValueError(
                "LEAKAGE RISK: Cannot fit calibration on test set. "
                "Temperature scaling must fit on VALIDATION set only. "
                "Test set must remain held-out for final evaluation."
            )
        
        logger.info(
            f"Fitting temperature scaler on {fit_set_name} set "
            f"({len(logits)} samples)"
        )
        
        logits_np = logits.flatten()
        labels_np = labels.flatten()

        # Grid search over temperature values to minimize ECE
        best_temp = 1.0
        best_ece = float('inf')

        for t in np.concatenate([
            np.arange(0.1, 1.0, 0.05),
            np.arange(1.0, 5.0, 0.1),
            np.arange(5.0, 20.0, 0.5),
        ]):
            scaled_probs = 1.0 / (1.0 + np.exp(-logits_np / max(t, 1e-6)))
            ece = compute_ece(scaled_probs, labels_np)
            if ece < best_ece:
                best_ece = ece
                best_temp = float(t)

        self.temperature.data = torch.tensor([best_temp], dtype=torch.float32)
        self._is_fitted = True

        logger.info(
            f"Temperature scaler fitted (ECE grid search): "
            f"temperature={best_temp:.4f}, ECE={best_ece:.6f}"
        )

        return best_temp
    
    def is_fitted(self) -> bool:
        """Check if scaler has been fitted."""
        return self._is_fitted


def optimize_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    target_sensitivity: float = 0.95,
    target_specificity: float = 0.90,
) -> Dict[str, float]:
    validator = ClinicalValidator(
        target_sensitivity=target_sensitivity,
        target_specificity=target_specificity,
    )
    results = validator.validate(labels, probs)
    thresholds = {
        "balanced": results["balanced_threshold"]["threshold"],
        "sensitivity": results["sensitivity_threshold"]["threshold"]
        if results["sensitivity_threshold"]["meets_requirement"]
        else results["balanced_threshold"]["threshold"],
        "specificity": results["specificity_threshold"]["threshold"]
        if results["specificity_threshold"]["meets_requirement"]
        else results["balanced_threshold"]["threshold"],
    }
    return thresholds


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error (ECE).
    """
    probs = probs.flatten()
    labels = labels.flatten()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i + 1])
        if not np.any(mask):
            continue
        bin_prob = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += np.abs(bin_acc - bin_prob) * (mask.mean())
    return float(ece)


def optimize_ensemble_weights(
    model_probs: list,
    labels: np.ndarray,
    metric: str = "auc",
    n_trials: int = 5000,
) -> np.ndarray:
    """
    Optimize ensemble weights on validation set to maximize AUC.

    Uses constrained random search: w_i >= 0, sum(w) = 1.

    CRITICAL: Must be called with VALIDATION set predictions only.
    Never use test set predictions (data leakage).

    Args:
        model_probs: List of arrays, each (N,) calibrated probabilities
            from one model on the validation set.
        labels: Ground truth labels (N,) from the validation set.
        metric: Optimization metric. 'auc' (default) or 'ece'.
        n_trials: Number of random weight samples to evaluate.

    Returns:
        Optimal weight vector (K,) summing to 1.
    """
    from sklearn.metrics import roc_auc_score

    K = len(model_probs)
    if K < 2:
        logger.warning("Only 1 model provided; returning weight [1.0]")
        return np.array([1.0])

    labels_flat = labels.flatten()
    probs_matrix = np.stack([p.flatten() for p in model_probs], axis=0)  # (K, N)

    best_score = -np.inf if metric == "auc" else np.inf
    best_weights = np.ones(K) / K  # default: equal

    rng = np.random.RandomState(42)

    for _ in range(n_trials):
        raw = rng.dirichlet(np.ones(K))
        ensemble_prob = raw @ probs_matrix  # (N,)

        if metric == "auc":
            try:
                score = roc_auc_score(labels_flat, ensemble_prob)
            except ValueError:
                continue
            if score > best_score:
                best_score = score
                best_weights = raw.copy()
        elif metric == "ece":
            score = compute_ece(ensemble_prob, labels_flat)
            if score < best_score:
                best_score = score
                best_weights = raw.copy()

    logger.info(
        f"Ensemble weight optimization ({n_trials} trials, metric={metric}): "
        f"best_score={best_score:.6f}, weights={best_weights.round(4).tolist()}"
    )
    return best_weights
