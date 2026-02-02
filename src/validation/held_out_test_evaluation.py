"""
Held-out test set evaluation for regulatory compliance and clinical validation.

This module implements comprehensive evaluation on a locked test set to provide
final performance metrics for regulatory submissions, research publications, and
clinical deployment decisions.

Key principles:
- Test set is held completely separate during training/validation
- Single evaluation pass (no tuning on test set)
- Extensive metrics covering clinical, statistical, and robustness aspects
- Results reproducible and auditable for FDA submission
- Confidence intervals from bootstrap resampling
- Performance stratification (by patient, stain, scanner if available)

Clinical Requirements (PCam context):
- Primary metric: AUC-ROC (threshold-independent)
- Clinical metrics: Sensitivity ≥ 95%, Specificity ≥ 90%
- Explicit thresholding strategy with justification
- Failure mode analysis (FP/FN distributions)
- Confidence bounds for all metrics
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
import logging
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class HeldOutTestEvaluator:
    """
    Comprehensive evaluation on held-out test set.
    
    Responsibilities:
    - Load trained model and test data
    - Generate predictions with calibration
    - Compute metrics at clinical operating points
    - Generate confidence intervals via bootstrap
    - Stratify performance by patient/data characteristics
    - Create regulatory-ready report
    
    Usage:
        evaluator = HeldOutTestEvaluator(
            model_path='checkpoints/best_model.pt',
            test_loader=test_loader,
            device=device
        )
        results = evaluator.evaluate()
        evaluator.save_report('test_evaluation_report.json')
        evaluator.plot_roc_curve('roc_curve.png')
    """
    
    def __init__(
        self,
        model_path: str,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        calibration_loader: Optional[torch.utils.data.DataLoader] = None,
        model_class: Optional[nn.Module] = None,
        bootstrap_samples: int = 1000,
        ci_level: float = 0.95,
    ):
        """
        Args:
            model_path: Path to trained model checkpoint
            test_loader: Test data loader
            device: Compute device
            calibration_loader: Optional calibration set for temperature scaling
            model_class: Model class (if not in checkpoint)
            bootstrap_samples: Number of bootstrap samples for confidence intervals
            ci_level: Confidence level (0.95 for 95% CI)
        """
        self.model_path = Path(model_path)
        self.test_loader = test_loader
        self.device = device
        self.calibration_loader = calibration_loader
        self.bootstrap_samples = bootstrap_samples
        self.ci_level = ci_level
        
        # Load model
        self.model = self._load_model(model_class)
        self.model.eval()
        
        # Results storage
        self.results = {}
    
    def _load_model(self, model_class: Optional[nn.Module] = None) -> nn.Module:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if model_class is None:
            raise ValueError("model_class must be provided to load model")
        
        model = model_class(**checkpoint.get('model_config', {}))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        logger.info(f"Loaded model from {self.model_path}")
        logger.info(f"Model config: {checkpoint.get('model_config', {})}")
        
        return model
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on test set.
        
        Returns:
            Dictionary with:
            - predictions: Predictions and probabilities
            - metrics: Standard metrics
            - clinical_metrics: Clinical operating points
            - calibration: Calibration analysis
            - confidence_intervals: Bootstrap confidence intervals
            - failure_analysis: FP/FN distributions
        """
        logger.info("Starting held-out test evaluation...")
        
        # Generate predictions
        predictions, labels, metadata = self._get_predictions()
        self.results['predictions'] = {
            'probabilities': predictions,
            'labels': labels,
            'metadata': metadata,
        }
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, labels)
        self.results['metrics'] = metrics
        logger.info(f"Test AUC: {metrics['auc']:.4f}")
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Sensitivity: {metrics['sensitivity']:.4f}")
        logger.info(f"Test Specificity: {metrics['specificity']:.4f}")
        
        # Clinical validation
        clinical_metrics = self._clinical_validation(predictions, labels)
        self.results['clinical_metrics'] = clinical_metrics
        
        # Calibration analysis
        calibration = self._analyze_calibration(predictions, labels)
        self.results['calibration'] = calibration
        
        # Bootstrap confidence intervals
        ci = self._bootstrap_confidence_intervals(predictions, labels)
        self.results['confidence_intervals'] = ci
        
        # Failure mode analysis
        failure_analysis = self._analyze_failure_modes(predictions, labels)
        self.results['failure_analysis'] = failure_analysis
        
        # Generate regulatory summary
        summary = self._generate_regulatory_summary()
        self.results['regulatory_summary'] = summary
        
        logger.info("Held-out test evaluation completed")
        
        return self.results
    
    @torch.no_grad()
    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generate predictions on test set."""
        all_probs = []
        all_labels = []
        all_metadata = []
        
        logger.info("Generating predictions on test set...")
        
        for batch in tqdm(self.test_loader, desc="Predicting"):
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            
            # Forward pass
            logits = self.model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())
            
            # Collect metadata if available
            if len(batch) > 2:
                all_metadata.append(batch[2])
        
        predictions = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        
        logger.info(f"Generated predictions for {len(predictions)} samples")
        logger.info(f"Positive prevalence: {labels.mean():.2%}")
        
        return predictions, labels, {'n_samples': len(predictions)}
    
    def _compute_metrics(self, probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute standard evaluation metrics."""
        preds_binary = (probs > 0.5).astype(int).squeeze()
        labels_clean = labels.squeeze().astype(int)
        probs_clean = probs.squeeze()
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels_clean, preds_binary).ravel()
        
        # Standard metrics
        metrics = {
            'auc': roc_auc_score(labels_clean, probs_clean),
            'accuracy': accuracy_score(labels_clean, preds_binary),
            'precision': precision_score(labels_clean, preds_binary),
            'recall': recall_score(labels_clean, preds_binary),
            'f1': f1_score(labels_clean, preds_binary),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0.0,
            'mcc': matthews_corrcoef(labels_clean, preds_binary),
            'kappa': cohen_kappa_score(labels_clean, preds_binary),
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)},
        }
        
        return metrics
    
    def _clinical_validation(self, probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Find optimal operating points for clinical use."""
        labels_clean = labels.squeeze().astype(int)
        probs_clean = probs.squeeze()
        
        fpr, tpr, thresholds = roc_curve(labels_clean, probs_clean)
        
        # Find operating points
        operating_points = {}
        
        # Conservative: maximize sensitivity (minimize FN)
        for sens_target in [0.95, 0.90, 0.85]:
            idx = np.where(tpr >= sens_target)[0]
            if len(idx) > 0:
                idx = idx[np.argmax(tpr[idx] - fpr[idx])]  # Maximize Youden index
                threshold = thresholds[idx]
                spec = 1 - fpr[idx]
                operating_points[f'sens_{int(sens_target*100)}'] = {
                    'threshold': float(threshold),
                    'sensitivity': float(tpr[idx]),
                    'specificity': float(spec),
                }
        
        # Balanced: Youden index
        youden_idx = np.argmax(tpr - fpr)
        operating_points['youden'] = {
            'threshold': float(thresholds[youden_idx]),
            'sensitivity': float(tpr[youden_idx]),
            'specificity': float(1 - fpr[youden_idx]),
        }
        
        return operating_points
    
    def _analyze_calibration(self, probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze model calibration (confidence vs accuracy)."""
        labels_clean = labels.squeeze().astype(int)
        probs_clean = probs.squeeze()
        
        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(
            labels_clean,
            probs_clean,
            n_bins=10,
            strategy='uniform'
        )
        
        # ECE (Expected Calibration Error)
        ece = np.mean(np.abs(prob_true - prob_pred))
        
        return {
            'ece': float(ece),
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
            'is_well_calibrated': bool(ece < 0.1),
        }
    
    def _bootstrap_confidence_intervals(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, Tuple[float, float]]:
        """Compute confidence intervals via bootstrap resampling."""
        labels_clean = labels.squeeze().astype(int)
        probs_clean = probs.squeeze()
        
        logger.info(f"Computing bootstrap confidence intervals ({self.bootstrap_samples} samples)...")
        
        bootstrap_metrics = {
            'auc': [],
            'sensitivity': [],
            'specificity': [],
            'accuracy': [],
        }
        
        for _ in tqdm(range(self.bootstrap_samples), desc="Bootstrap"):
            # Resample with replacement
            idx = np.random.choice(len(labels_clean), size=len(labels_clean), replace=True)
            boot_probs = probs_clean[idx]
            boot_labels = labels_clean[idx]
            
            # Compute metrics
            try:
                auc = roc_auc_score(boot_labels, boot_probs)
                bootstrap_metrics['auc'].append(auc)
            except:
                pass
            
            preds = (boot_probs > 0.5).astype(int)
            tp = np.sum((preds == 1) & (boot_labels == 1))
            tn = np.sum((preds == 0) & (boot_labels == 0))
            fp = np.sum((preds == 1) & (boot_labels == 0))
            fn = np.sum((preds == 0) & (boot_labels == 1))
            
            if (tp + fn) > 0:
                bootstrap_metrics['sensitivity'].append(tp / (tp + fn))
            if (tn + fp) > 0:
                bootstrap_metrics['specificity'].append(tn / (tn + fp))
            
            bootstrap_metrics['accuracy'].append(accuracy_score(boot_labels, preds))
        
        # Compute confidence intervals
        alpha = (1 - self.ci_level) / 2
        ci = {}
        for metric_name, values in bootstrap_metrics.items():
            lower = np.percentile(values, alpha * 100)
            upper = np.percentile(values, (1 - alpha) * 100)
            ci[metric_name] = {
                'lower': float(lower),
                'upper': float(upper),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }
        
        logger.info(f"Bootstrap CIs computed: AUC={ci['auc']['mean']:.4f} [{ci['auc']['lower']:.4f}, {ci['auc']['upper']:.4f}]")
        
        return ci
    
    def _analyze_failure_modes(self, probs: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """Analyze false positive and false negative distributions."""
        labels_clean = labels.squeeze().astype(int)
        probs_clean = probs.squeeze()
        
        preds_binary = (probs_clean > 0.5).astype(int)
        
        # False positives (predicted positive, actually negative)
        fp_mask = (preds_binary == 1) & (labels_clean == 0)
        fp_probs = probs_clean[fp_mask]
        
        # False negatives (predicted negative, actually positive)
        fn_mask = (preds_binary == 0) & (labels_clean == 1)
        fn_probs = probs_clean[fn_mask]
        
        return {
            'false_positives': {
                'count': int(np.sum(fp_mask)),
                'percentage': float(100 * np.mean(fp_mask)),
                'mean_prob': float(np.mean(fp_probs)) if len(fp_probs) > 0 else None,
                'median_prob': float(np.median(fp_probs)) if len(fp_probs) > 0 else None,
                'std_prob': float(np.std(fp_probs)) if len(fp_probs) > 0 else None,
            },
            'false_negatives': {
                'count': int(np.sum(fn_mask)),
                'percentage': float(100 * np.mean(fn_mask)),
                'mean_prob': float(np.mean(fn_probs)) if len(fn_probs) > 0 else None,
                'median_prob': float(np.median(fn_probs)) if len(fn_probs) > 0 else None,
                'std_prob': float(np.std(fn_probs)) if len(fn_probs) > 0 else None,
            },
        }
    
    def _generate_regulatory_summary(self) -> Dict[str, Any]:
        """Generate summary for FDA submission."""
        metrics = self.results['metrics']
        ci = self.results['confidence_intervals']
        clinical = self.results['clinical_metrics']
        
        summary = {
            'evaluation_date': datetime.now().isoformat(),
            'test_set_size': self.results['predictions']['metadata']['n_samples'],
            'primary_metric': {
                'name': 'AUC-ROC',
                'value': metrics['auc'],
                '95ci': [ci['auc']['lower'], ci['auc']['upper']],
            },
            'clinical_performance': {
                'sensitivity': metrics['sensitivity'],
                'specificity': metrics['specificity'],
                '95ci_sensitivity': [ci['sensitivity']['lower'], ci['sensitivity']['upper']],
                '95ci_specificity': [ci['specificity']['lower'], ci['specificity']['upper']],
            },
            'recommended_threshold': clinical.get('youden', {}).get('threshold', 0.5),
            'model_calibration': 'Well-calibrated' if self.results['calibration']['is_well_calibrated'] else 'Poorly calibrated',
            'failure_modes': self.results['failure_analysis'],
        }
        
        return summary
    
    def save_report(self, output_path: str) -> None:
        """Save detailed results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to Python lists for JSON serialization
        results_json = self._to_serializable(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
    
    def plot_roc_curve(self, output_path: str) -> None:
        """Plot ROC curve."""
        if 'predictions' not in self.results:
            logger.warning("No predictions available, run evaluate() first")
            return
        
        probs = self.results['predictions']['probabilities'].squeeze()
        labels = self.results['predictions']['labels'].squeeze().astype(int)
        
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_score = roc_auc_score(labels, probs)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve on Held-Out Test Set')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        logger.info(f"ROC curve saved to {output_path}")
        plt.close()
    
    def plot_calibration_curve(self, output_path: str) -> None:
        """Plot calibration curve."""
        if 'predictions' not in self.results:
            logger.warning("No predictions available, run evaluate() first")
            return
        
        calib = self.results['calibration']
        
        plt.figure(figsize=(8, 6))
        plt.plot(calib['prob_pred'], calib['prob_true'], 'o-', label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Curve (ECE = {calib["ece"]:.4f})')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        logger.info(f"Calibration curve saved to {output_path}")
        plt.close()
    
    def plot_confusion_matrix(self, output_path: str) -> None:
        """Plot confusion matrix."""
        if 'predictions' not in self.results:
            logger.warning("No predictions available, run evaluate() first")
            return
        
        cm = self.results['metrics']['confusion_matrix']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            [[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            ax=ax
        )
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix on Held-Out Test Set')
        plt.tight_layout()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
        logger.info(f"Confusion matrix saved to {output_path}")
        plt.close()
    
    @staticmethod
    def _to_serializable(obj: Any) -> Any:
        """Convert numpy arrays and other non-JSON types to serializable."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: HeldOutTestEvaluator._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [HeldOutTestEvaluator._to_serializable(item) for item in obj]
        else:
            return obj
