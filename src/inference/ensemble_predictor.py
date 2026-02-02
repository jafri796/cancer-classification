"""
Ensemble predictor with uncertainty-aware aggregation and strategic diversity.

Ensemble methods improve robustness by:
- Reducing correlated errors through model diversity
- Providing uncertainty estimates for clinical decision-making
- Improving generalization to unseen staining variations
- Enabling multi-site deployment with enhanced confidence
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Union
import yaml
import numpy as np
import torch
import logging
import time

from src.data.preprocessing import get_transforms, TestTimeAugmentation
from src.inference.model_registry import load_pretrained_model
from src.inference.predictor import PCamPredictor
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models for robust predictions.
    
    Ensemble Strategy (Soft Voting with Weighted Averaging):
    - Each model provides a probability estimate
    - Predictions are averaged with learned weights
    - Weights can be adjusted based on model performance
    - Uncertainty is computed as standard deviation across models
    
    Medical Rationale:
    - No single model is perfect; ensembles reduce error
    - Different architectures capture different morphological features
    - Uncertainty estimates guide clinical decision-making
    - Diversity (ResNet + EfficientNet + ViT) ensures robustness
    
    Clinical Requirements:
    - All configurations logged for reproducibility
    - Uncertainty quantification for confidence-aware decisions
    - Fast inference (<200ms for ensemble)
    - Deterministic results for regulatory compliance
    
    Args:
        models: List of model configurations (path, weight, threshold, etc.)
        model_config_path: Path to model architecture config
        data_config_path: Path to data preprocessing config
        device: Compute device (cuda/cpu)
        threshold: Classification threshold
        use_tta: Enable test-time augmentation
    """
    
    def __init__(
        self,
        models: List[Dict],
        model_config_path: Union[str, Path],
        data_config_path: Union[str, Path],
        device: str = "cuda",
        threshold: float = 0.5,
        use_tta: bool = False,
    ):
        # Ensure reproducibility
        set_seed(42, deterministic=True, benchmark=False)
        
        self.device = device
        self.threshold = threshold
        self.use_tta = use_tta
        self.model_config_path = model_config_path
        self.data_config_path = data_config_path
        self.ensemble_times = []  # Track ensemble latency

        self.predictors = []
        self.weights = []
        
        logger.info(f"Initializing ensemble with {len(models)} models")

        for i, entry in enumerate(models):
            try:
                predictor = PCamPredictor(
                    model_path=entry.get("model_path", "models/final_model.pt"),
                    model_config_path=model_config_path,
                    data_config_path=data_config_path,
                    device=device,
                    threshold=entry.get("threshold", threshold),
                    use_tta=use_tta,
                    pretrained_id=entry.get("pretrained_id"),
                    registry_path=entry.get("registry_path"),
                    calibration_path=entry.get("calibration_path"),
                )
                self.predictors.append(predictor)
                weight = float(entry.get("weight", 1.0))
                self.weights.append(weight)
                logger.info(f"Loaded model {i+1}/{len(models)}: weight={weight:.3f}, threshold={entry.get('threshold', threshold)}")
            except Exception as e:
                logger.error(f"Failed to load model {i+1}: {str(e)}")
                raise

        # Normalize weights to sum to 1.0
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        
        logger.info(f"Ensemble weights (normalized): {[f'{w:.3f}' for w in self.weights]}")
        logger.info(f"Ensemble initialized successfully with {len(self.predictors)} models")

    def predict(self, image) -> Dict[str, object]:
        """
        Ensemble prediction with uncertainty estimation.
        
        Strategy:
        1. Get probability from each model
        2. Compute weighted average (soft voting)
        3. Compute uncertainty (standard deviation)
        4. Apply threshold for binary classification
        
        Returns:
            Dict with: probability, label, threshold, uncertainty, 
                      ensemble_size, model_probabilities, weights, latency_ms
        """
        start_time = time.perf_counter()
        
        probs = []
        for i, predictor in enumerate(self.predictors):
            try:
                result = predictor.predict(image)
                probs.append(result["probability"])
                logger.debug(f"Model {i+1} probability: {result['probability']:.4f}")
            except Exception as e:
                logger.error(f"Model {i+1} prediction failed: {str(e)}")
                raise
        
        probs = np.array(probs)
        
        # Weighted average (soft voting)
        weighted_prob = float(np.sum(probs * np.array(self.weights)))
        
        # Uncertainty (std dev across ensemble members)
        uncertainty = float(np.std(probs))
        
        # Binary classification
        label = int(weighted_prob >= self.threshold)
        
        # Track ensemble latency
        ensemble_time_ms = (time.perf_counter() - start_time) * 1000
        self.ensemble_times.append(ensemble_time_ms)
        
        if ensemble_time_ms > 200:
            logger.warning(f"Ensemble inference time {ensemble_time_ms:.1f}ms exceeds 200ms target")
        
        return {
            "probability": weighted_prob,
            "label": label,
            "threshold": self.threshold,
            "uncertainty": uncertainty,
            "ensemble_size": len(self.predictors),
            "model_probabilities": probs.tolist(),
            "weights": [f"{w:.4f}" for w in self.weights],
            "latency_ms": f"{ensemble_time_ms:.2f}",
        }

