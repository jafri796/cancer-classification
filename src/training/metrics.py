"""
Medical-grade metrics for binary classification in histopathology.

Implements clinically-relevant metrics:
- AUC-ROC: Threshold-independent performance
- Sensitivity (Recall): Percentage of center-region tumors correctly identified
- Specificity: Percentage of healthy tissue correctly identified
- PPV/NPV: Predictive values for clinical decision-making
- Cohen's Kappa: Agreement with pathologist labels
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
)
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class MedicalMetrics:
    """
    Comprehensive medical metrics tracker.
    
    Computes:
    - Standard ML metrics (accuracy, precision, recall, F1, AUC)
    - Clinical metrics (sensitivity, specificity, PPV, NPV)
    - Inter-rater agreement (Cohen's Kappa)
    
    Design:
    - Accumulates predictions/labels across batches
    - Computes metrics at epoch end for stability
    - GPU-accelerated where possible
    
    Args:
        threshold: Decision threshold for binary classification
        device: Compute device
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        device: torch.device = torch.device('cpu'),
    ):
        self.threshold = threshold
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions and labels."""
        self.predictions = []
        self.probabilities = []
        self.labels = []
    
    def update(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Update metrics with batch predictions.
        
        Args:
            probs: Predicted probabilities (B, 1)
            labels: Ground truth labels (B, 1)
        """
        # Move to CPU and convert to numpy
        probs_np = probs.detach().cpu().numpy().flatten()
        labels_np = labels.detach().cpu().numpy().flatten()
        
        # Store
        self.probabilities.append(probs_np)
        self.labels.append(labels_np)
        
        # Compute predictions
        preds_np = (probs_np >= self.threshold).astype(int)
        self.predictions.append(preds_np)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated data.
        
        Returns:
            Dictionary of metric name -> value
        """
        # Concatenate all batches
        y_true = np.concatenate(self.labels)
        y_pred = np.concatenate(self.predictions)
        y_prob = np.concatenate(self.probabilities)
        
        # Validate data
        assert len(y_true) == len(y_pred) == len(y_prob), "Length mismatch"
        assert y_true.min() >= 0 and y_true.max() <= 1, "Invalid labels"
        assert y_prob.min() >= 0 and y_prob.max() <= 1, "Invalid probabilities"
        
        # Compute metrics
        metrics = {}
        
        # Standard ML metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC-ROC (threshold-independent)
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError as e:
            logger.warning(f"Could not compute AUC: {e}")
            metrics['auc'] = 0.0
        
        # Clinical metrics from confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        # Sensitivity (True Positive Rate, Recall for positive class)
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity (True Negative Rate)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Positive Predictive Value (Precision for positive class)
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Negative Predictive Value
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        
        # False Positive Rate
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # False Negative Rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Cohen's Kappa (inter-rater agreement)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Confusion matrix elements (for detailed analysis)
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Sample counts
        metrics['total_samples'] = len(y_true)
        metrics['positive_samples'] = int(y_true.sum())
        metrics['negative_samples'] = int(len(y_true) - y_true.sum())
        
        return metrics
    
    def compute_at_threshold(self, threshold: float) -> Dict[str, float]:
        """
        Compute metrics at a specific threshold.
        
        Useful for clinical validation at multiple operating points.
        
        Args:
            threshold: Decision threshold
            
        Returns:
            Dictionary of metrics
        """
        # Temporarily change threshold
        original_threshold = self.threshold
        self.threshold = threshold
        
        # Recompute predictions
        y_prob = np.concatenate(self.probabilities)
        self.predictions = [(y_prob >= threshold).astype(int)]
        
        # Compute metrics
        metrics = self.compute()
        
        # Restore original threshold and predictions
        self.threshold = original_threshold
        
        return metrics
    
    def get_optimal_threshold(
        self,
        criterion: str = 'youden'
    ) -> float:
        """
        Find optimal decision threshold.
        
        Args:
            criterion: 'youden' (maximize sensitivity+specificity-1) or
                      'f1' (maximize F1 score)
        
        Returns:
            Optimal threshold value
        """
        from sklearn.metrics import roc_curve
        
        y_true = np.concatenate(self.labels)
        y_prob = np.concatenate(self.probabilities)
        
        # Get all possible thresholds
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        if criterion == 'youden':
            # Youden's J statistic: sensitivity + specificity - 1
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
        elif criterion == 'f1':
            # Maximize F1 score
            f1_scores = []
            for threshold in thresholds:
                y_pred = (y_prob >= threshold).astype(int)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                f1_scores.append(f1)
            optimal_idx = np.argmax(f1_scores)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        optimal_threshold = float(thresholds[optimal_idx])
        
        logger.info(
            f"Optimal threshold ({criterion}): {optimal_threshold:.4f} "
            f"(Sensitivity: {tpr[optimal_idx]:.4f}, "
            f"Specificity: {1-fpr[optimal_idx]:.4f})"
        )
        
        return optimal_threshold
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix as 2x2 array."""
        y_true = np.concatenate(self.labels)
        y_pred = np.concatenate(self.predictions)
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(self) -> str:
        """Get detailed classification report."""
        from sklearn.metrics import classification_report
        
        y_true = np.concatenate(self.labels)
        y_pred = np.concatenate(self.predictions)
        
        return classification_report(
            y_true,
            y_pred,
            target_names=['Negative', 'Positive'],
            digits=4
        )


class ClinicalValidator:
    """
    Clinical validation analyzer.
    
    Analyzes model performance across multiple thresholds to find
    clinically acceptable operating points.
    
    Medical Requirements:
    - Sensitivity ≥ 95% (minimize false negatives)
    - Specificity ≥ 90% (minimize false positives)
    """
    
    def __init__(
        self,
        target_sensitivity: float = 0.95,
        target_specificity: float = 0.90,
    ):
        self.target_sensitivity = target_sensitivity
        self.target_specificity = target_specificity
    
    def validate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, any]:
        """
        Perform clinical validation.
        
        Args:
            y_true: Ground truth labels
            y_prob: Predicted probabilities
            
        Returns:
            Clinical validation results
        """
        from sklearn.metrics import roc_curve
        
        # Get ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        
        # Find thresholds meeting clinical requirements
        sensitivity = tpr
        specificity = 1 - fpr
        
        # Threshold for target sensitivity
        sens_mask = sensitivity >= self.target_sensitivity
        if sens_mask.any():
            sens_threshold = thresholds[sens_mask][-1]
            sens_specificity = specificity[sens_mask][-1]
        else:
            sens_threshold = None
            sens_specificity = None
        
        # Threshold for target specificity
        spec_mask = specificity >= self.target_specificity
        if spec_mask.any():
            spec_threshold = thresholds[spec_mask][0]
            spec_sensitivity = sensitivity[spec_mask][0]
        else:
            spec_threshold = None
            spec_sensitivity = None
        
        # Best balanced threshold (Youden's index)
        youden_scores = sensitivity + specificity - 1
        best_idx = np.argmax(youden_scores)
        balanced_threshold = thresholds[best_idx]
        balanced_sensitivity = sensitivity[best_idx]
        balanced_specificity = specificity[best_idx]
        
        results = {
            'target_sensitivity': self.target_sensitivity,
            'target_specificity': self.target_specificity,
            'sensitivity_threshold': {
                'threshold': float(sens_threshold) if sens_threshold else None,
                'sensitivity': float(self.target_sensitivity) if sens_threshold else None,
                'specificity': float(sens_specificity) if sens_threshold else None,
                'meets_requirement': sens_threshold is not None,
            },
            'specificity_threshold': {
                'threshold': float(spec_threshold) if spec_threshold else None,
                'sensitivity': float(spec_sensitivity) if spec_threshold else None,
                'specificity': float(self.target_specificity) if spec_threshold else None,
                'meets_requirement': spec_threshold is not None,
            },
            'balanced_threshold': {
                'threshold': float(balanced_threshold),
                'sensitivity': float(balanced_sensitivity),
                'specificity': float(balanced_specificity),
            },
        }
        
        return results
    
    def evaluate_at_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
    ) -> Dict[str, float]:
        """Evaluate metrics at specific threshold."""
        y_pred = (y_prob >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        return {
            'threshold': threshold,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        }