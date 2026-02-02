"""
Clinical validation utilities for PCam center-region detection.
"""

from typing import Dict
import logging

from src.training.metrics import ClinicalValidator

logger = logging.getLogger(__name__)


def run_clinical_validation(
    y_true,
    y_prob,
    target_sensitivity: float = 0.95,
    target_specificity: float = 0.90,
) -> Dict[str, object]:
    """Run clinical validation and return threshold analysis."""
    validator = ClinicalValidator(
        target_sensitivity=target_sensitivity,
        target_specificity=target_specificity,
    )
    results = validator.validate(y_true, y_prob)
    logger.info("Clinical validation complete")
    return results