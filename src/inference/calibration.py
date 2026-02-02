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
        
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.1, max_iter=max_iter)
        loss_fn = nn.BCEWithLogitsLoss()

        def _closure():
            optimizer.zero_grad()
            loss = loss_fn(self.forward(logits_t), labels_t)
            loss.backward()
            return loss

        optimizer.step(_closure)
        
        temperature_val = float(self.temperature.item())
        self._is_fitted = True
        
        logger.info(
            f"Temperature scaler fitted: temperature={temperature_val:.4f}"
        )
        
        return temperature_val
    
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
