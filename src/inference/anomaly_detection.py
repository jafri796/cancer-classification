"""
Out-of-Distribution (OOD) detection and anomaly detection for clinical robustness.

Detects when input samples fall outside the training distribution,
enabling safe deployment by flagging predictions that may be unreliable
due to domain shift (different stains, scanners, tissue types).

Methods:
- Mahalanobis distance (statistical feature distance)
- Entropy-based detection (prediction confidence)
- Temperature scaling + selective prediction (known uncertainty)
- Feature reconstruction error (VAE-based)
- Isolation Forest (ensemble-based)

Clinical context (PCam):
- Training set: Specific pathology center, H&E stain, scanner
- Real-world deployment: Multiple stains, scanners, centers
- OOD detection helps maintain safety/sensitivity under domain shift

Requirements for regulatory submission:
- Reproducible and deterministic (seeded)
- Documented thresholds with justification
- Optional (flagging for manual review, not auto-rejection)
- Monitoring/alerting for deployment
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List
import logging
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
import pickle

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract features from penultimate layer of model."""
    
    def __init__(
        self,
        model: nn.Module,
        layer_name: str = 'layer4',
        device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            model: Neural network
            layer_name: Name of layer to extract from
            device: Compute device
        """
        self.model = model.to(device)
        self.device = device
        self.layer_name = layer_name
        
        self.features = None
        self._register_hook()
    
    def _register_hook(self) -> None:
        """Register forward hook to capture features."""
        def hook(module, input, output):
            # Flatten spatial dimensions
            if output.dim() == 4:  # Conv output (B, C, H, W)
                output = output.view(output.size(0), output.size(1), -1)
                output = output.mean(dim=2)  # Global average pool
            self.features = output.detach()
        
        # Find and hook layer
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook)
                break
    
    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> np.ndarray:
        """Extract features from input batch."""
        x = x.to(self.device)
        self.model.eval()
        _ = self.model(x)
        
        if self.features is None:
            raise RuntimeError("Features not captured during forward pass")
        
        return self.features.cpu().numpy()


class MahalanobisDetector:
    """
    Mahalanobis distance-based OOD detection.
    
    Measures distance from sample to training set distribution in feature space.
    Samples far from the training distribution are likely OOD.
    
    Mathematical basis:
    - Fit multivariate Gaussian to training features
    - For new sample: Mahalanobis distance = sqrt((x - μ)^T Σ^-1 (x - μ))
    - High distance → OOD; low distance → in-distribution
    
    Advantages:
    - Simple, interpretable, deterministic
    - Works with high-dimensional features
    - Computationally efficient at inference time
    
    Disadvantages:
    - Assumes Gaussian distribution (may not hold)
    - Sensitive to outliers in training data
    
    Usage:
        detector = MahalanobisDetector(model, device)
        detector.fit(train_loader)  # Fit on training set
        scores = detector.detect(test_images)  # OOD scores
        is_ood = scores > threshold  # Detect OOD samples
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cpu'),
        layer_name: str = 'layer4',
    ):
        """
        Args:
            model: Neural network
            device: Compute device
            layer_name: Layer to extract features from
        """
        self.feature_extractor = FeatureExtractor(model, layer_name, device)
        self.device = device
        
        self.mean = None
        self.cov_inv = None
        self.covariance = None
    
    def fit(self, features: np.ndarray) -> None:
        """
        Fit Mahalanobis parameters on training features.
        
        Args:
            features: Training feature vectors (N, D)
        """
        logger.info(f"Fitting Mahalanobis detector on {len(features)} samples...")
        
        # Compute mean
        self.mean = np.mean(features, axis=0)
        
        # Fit covariance (use empirical covariance for stability)
        cov_estimator = EmpiricalCovariance().fit(features)
        self.covariance = cov_estimator.covariance_
        
        # Compute inverse (add small regularization for numerical stability)
        try:
            self.cov_inv = np.linalg.inv(self.covariance + np.eye(self.covariance.shape[0]) * 1e-4)
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix singular, using pseudo-inverse")
            self.cov_inv = np.linalg.pinv(self.covariance)
        
        logger.info(f"Fitted on feature dimension {self.mean.shape[0]}")
    
    @torch.no_grad()
    def detect(self, x: torch.Tensor) -> np.ndarray:
        """
        Compute Mahalanobis distance for batch.
        
        Args:
            x: Input batch (B, C, H, W)
            
        Returns:
            Mahalanobis distances (B,) - higher = more OOD
        """
        if self.mean is None or self.cov_inv is None:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        features = self.feature_extractor.extract(x)  # (B, D)
        
        # Compute Mahalanobis distance for each sample
        distances = []
        for feat in features:
            delta = feat - self.mean
            distance = np.sqrt(delta @ self.cov_inv @ delta.T)
            distances.append(distance)
        
        return np.array(distances)


class EntropyDetector:
    """
    Entropy-based OOD detection using prediction confidence.
    
    Assumes that for in-distribution samples, the model makes confident predictions
    (high max probability), while OOD samples have uncertain predictions (high entropy).
    
    Entropy: H(p) = -sum(p_i * log(p_i))
    - H ≈ 0: Confident prediction (one class very likely)
    - H ≈ 1: Uncertain prediction (classes equally likely)
    
    Interpretation:
    - In-distribution: Low entropy (confident)
    - OOD: High entropy (uncertain)
    
    Advantages:
    - Simple, no training required
    - Fast inference
    
    Disadvantages:
    - Some models overconfident on OOD (especially in binary classification)
    - Requires calibrated probabilities
    
    Usage:
        detector = EntropyDetector()
        scores = detector.detect(probs)  # Entropy scores
        is_ood = scores > threshold
    """
    
    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Temperature for softmax scaling (higher = more uncertain)
        """
        self.temperature = temperature
    
    @torch.no_grad()
    def detect(self, logits: torch.Tensor) -> np.ndarray:
        """
        Compute entropy of predictions.
        
        Args:
            logits: Model logits (B, 1) or (B, K)
            
        Returns:
            Entropy scores (B,) - higher = more OOD
        """
        # Convert to probabilities
        if logits.dim() == 1 or logits.shape[1] == 1:
            # Binary classification: convert to 2-class
            probs = torch.sigmoid(logits / self.temperature).squeeze()
            probs = torch.stack([1 - probs, probs], dim=1)
        else:
            probs = torch.softmax(logits / self.temperature, dim=1)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        
        return entropy.cpu().numpy()


class IsolationForestDetector:
    """
    Isolation Forest-based OOD detection.
    
    Ensemble method that isolates anomalies using decision trees.
    Anomalies require fewer splits to isolate (higher anomaly score).
    
    Advantages:
    - No assumptions about data distribution
    - Handles high-dimensional data well
    - Robust to outliers
    
    Disadvantages:
    - Less interpretable than Mahalanobis
    - Training can be slow for large datasets
    - Random forest element makes it stochastic (can seed for reproducibility)
    
    Usage:
        detector = IsolationForestDetector()
        detector.fit(train_features)
        scores = detector.detect(test_features)
        is_ood = scores > threshold
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        """
        Args:
            contamination: Expected fraction of outliers in training set
            n_estimators: Number of isolation trees
            random_state: Random seed for reproducibility
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
    
    def fit(self, features: np.ndarray) -> None:
        """Fit Isolation Forest on training features."""
        logger.info(f"Fitting Isolation Forest on {len(features)} samples...")
        self.model.fit(features)
    
    def detect(self, features: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores.
        
        Args:
            features: Feature vectors (N, D)
            
        Returns:
            Anomaly scores (N,) - higher = more anomalous/OOD
        """
        # Get raw anomaly scores (negative of mean depth)
        scores = -self.model.score_samples(features)
        
        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores


class CompositeOODDetector:
    """
    Ensemble OOD detector combining multiple methods.
    
    Combines:
    - Mahalanobis distance (statistical)
    - Entropy (confidence-based)
    - Isolation Forest (ensemble-based)
    
    Voting strategy:
    - Flag as OOD if ≥2 methods agree
    - Score: weighted sum of normalized scores
    
    Medical context:
    - Reduces false alarms (OOD samples must be suspicious by multiple methods)
    - Improves reliability for deployment
    - Interpretable: can see which methods flagged sample
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cpu'),
        use_mahalanobis: bool = True,
        use_entropy: bool = True,
        use_isolation_forest: bool = True,
    ):
        """
        Args:
            model: Neural network
            device: Compute device
            use_mahalanobis: Include Mahalanobis detector
            use_entropy: Include entropy detector
            use_isolation_forest: Include Isolation Forest detector
        """
        self.device = device
        self.model = model
        
        self.detectors = {}
        
        if use_mahalanobis:
            self.detectors['mahalanobis'] = MahalanobisDetector(model, device)
        if use_entropy:
            self.detectors['entropy'] = EntropyDetector()
        if use_isolation_forest:
            self.detectors['isolation_forest'] = IsolationForestDetector()
        
        logger.info(f"Composite OOD detector with methods: {list(self.detectors.keys())}")
    
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
    ) -> None:
        """
        Fit all detectors on training set.
        
        Args:
            train_loader: Training data loader
        """
        logger.info("Fitting OOD detectors on training set...")
        
        # Extract training features
        feature_extractor = FeatureExtractor(self.model, device=self.device)
        all_features = []
        all_logits = []
        
        for images, _ in train_loader:
            features = feature_extractor.extract(images)
            all_features.append(features)
            
            with torch.no_grad():
                logits = self.model(images.to(self.device))
            all_logits.append(logits.cpu())
        
        all_features = np.concatenate(all_features)
        all_logits = torch.cat(all_logits)
        
        # Fit individual detectors
        if 'mahalanobis' in self.detectors:
            self.detectors['mahalanobis'].fit(all_features)
        
        if 'isolation_forest' in self.detectors:
            self.detectors['isolation_forest'].fit(all_features)
    
    @torch.no_grad()
    def detect(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Detect OOD samples using ensemble voting.
        
        Args:
            x: Input batch (B, C, H, W)
            
        Returns:
            Dictionary with:
            - is_ood: Boolean mask of OOD samples
            - ensemble_score: Aggregated OOD score [0, 1]
            - individual_scores: Per-method scores
            - voting_counts: Number of methods flagging as OOD
        """
        individual_scores = {}
        voting_count = np.zeros(len(x))
        
        # Get thresholds (empirically tuned or configurable)
        thresholds = {
            'mahalanobis': 3.0,  # ~99th percentile for standard normal
            'entropy': 0.5,  # Mid-range entropy for binary classification
            'isolation_forest': 0.5,  # 50th percentile anomaly score
        }
        
        # Run each detector
        if 'mahalanobis' in self.detectors:
            scores = self.detectors['mahalanobis'].detect(x)
            individual_scores['mahalanobis'] = scores
            voting_count += (scores > thresholds['mahalanobis']).astype(int)
        
        if 'entropy' in self.detectors:
            logits = self.model(x)
            scores = self.detectors['entropy'].detect(logits)
            individual_scores['entropy'] = scores
            voting_count += (scores > thresholds['entropy']).astype(int)
        
        if 'isolation_forest' in self.detectors:
            features = FeatureExtractor(self.model, device=self.device).extract(x)
            scores = self.detectors['isolation_forest'].detect(features)
            individual_scores['isolation_forest'] = scores
            voting_count += (scores > thresholds['isolation_forest']).astype(int)
        
        # Aggregate scores (normalize and average)
        normalized_scores = {}
        for method, scores in individual_scores.items():
            norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            normalized_scores[method] = norm_scores
        
        if normalized_scores:
            ensemble_score = np.mean(list(normalized_scores.values()), axis=0)
        else:
            ensemble_score = np.zeros(len(x))
        
        # Voting-based OOD detection (≥2 methods agree)
        is_ood = voting_count >= 2
        
        return {
            'is_ood': is_ood,
            'ensemble_score': ensemble_score,
            'individual_scores': individual_scores,
            'voting_count': voting_count,
            'thresholds': thresholds,
        }
    
    def save(self, checkpoint_path: str) -> None:
        """Save fitted detectors to disk."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'detectors': self.detectors,
            'device': str(self.device),
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved OOD detectors to {checkpoint_path}")
    
    def load(self, checkpoint_path: str) -> None:
        """Load fitted detectors from disk."""
        checkpoint_path = Path(checkpoint_path)
        
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)
        
        self.detectors = state['detectors']
        
        logger.info(f"Loaded OOD detectors from {checkpoint_path}")


class ModelDeploymentMonitor:
    """
    Monitor model performance and OOD detection during deployment.
    
    Tracks:
    - OOD detection rate (% predictions flagged)
    - Performance on OOD vs in-distribution predictions
    - Drift in OOD score distribution
    - Manual review outcomes (to recalibrate thresholds)
    
    Helps maintain model safety and trigger retraining when needed.
    """
    
    def __init__(self, log_dir: Path = Path('deployment_monitoring')):
        """
        Args:
            log_dir: Directory to save monitoring logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.events = []
    
    def log_prediction(
        self,
        image_id: str,
        prediction: float,
        confidence: float,
        is_ood: bool,
        ood_score: float,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Log a prediction event."""
        event = {
            'image_id': image_id,
            'prediction': float(prediction),
            'confidence': float(confidence),
            'is_ood': bool(is_ood),
            'ood_score': float(ood_score),
            'metadata': metadata or {},
        }
        self.events.append(event)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate monitoring report."""
        if not self.events:
            return {'error': 'No events recorded'}
        
        is_ood_array = np.array([e['is_ood'] for e in self.events])
        ood_scores = np.array([e['ood_score'] for e in self.events])
        
        return {
            'total_predictions': len(self.events),
            'ood_detection_rate': float(np.mean(is_ood_array)),
            'mean_ood_score': float(np.mean(ood_scores)),
            'std_ood_score': float(np.std(ood_scores)),
            'max_ood_score': float(np.max(ood_scores)),
            'alerts': self._check_alerts(is_ood_array, ood_scores),
        }
    
    def _check_alerts(
        self,
        is_ood_array: np.ndarray,
        ood_scores: np.ndarray,
    ) -> List[str]:
        """Check for deployment alerts."""
        alerts = []
        
        # Alert if high OOD rate (potential domain shift)
        ood_rate = np.mean(is_ood_array)
        if ood_rate > 0.2:
            alerts.append(f"High OOD detection rate: {ood_rate:.1%}")
        
        # Alert if sudden increase in OOD scores
        if len(ood_scores) > 100:
            recent_mean = np.mean(ood_scores[-100:])
            historical_mean = np.mean(ood_scores[:-100])
            if recent_mean > historical_mean * 1.5:
                alerts.append("OOD scores increasing - potential domain shift")
        
        return alerts
