"""
MLOps experiment tracking for reproducibility and compliance.

Integrates with MLflow for:
- Experiment and run tracking
- Parameter logging
- Metric tracking
- Model versioning
- Artifact management
"""

import mlflow
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracker for clinical-grade reproducibility.
    
    Clinical Requirements:
    - All hyperparameters logged for audit trail
    - All metrics tracked at each epoch
    - Model artifacts versioned and tracked
    - Experiment metadata preserved
    """
    
    def __init__(self, experiment_name: str = "pcam_classification", tracking_uri: str = None):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Experiment name in MLflow
            tracking_uri: MLflow tracking server URI (default: local)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.experiment_name = experiment_name
        logger.info(f"MLflow tracker initialized for experiment: {experiment_name}")
    
    def start_run(self, run_name: str = None, tags: Dict[str, str] = None):
        """Start a new MLflow run."""
        mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)
        logger.info(f"Started MLflow run: {run_name}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters (hyperparameters, config, etc.)."""
        mlflow.log_params(params)
        logger.info(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics (loss, accuracy, AUC, etc.)."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        logger.debug(f"Logged metrics at step {step}")
    
    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        """Log an artifact (file, model, etc.)."""
        mlflow.log_artifact(artifact_path, artifact_name)
        logger.info(f"Logged artifact: {artifact_path}")
    
    def log_model(self, model_path: str, model_name: str = "pcam_model"):
        """Log a PyTorch model."""
        mlflow.pytorch.log_model(model_path, model_name)
        logger.info(f"Logged PyTorch model: {model_name}")
    
    def end_run(self):
        """End the current run."""
        mlflow.end_run()
        logger.info("Ended MLflow run")
    
    def log_config(self, config: Dict[str, Any], config_name: str = "config.json"):
        """Log entire configuration as JSON artifact."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            temp_path = f.name
        
        mlflow.log_artifact(temp_path, "configs")
        Path(temp_path).unlink()
        logger.info("Logged configuration JSON")


def log_experiment(config: Dict[str, Any], metrics: Dict[str, float], 
                  experiment_name: str = "pcam_classification"):
    """
    Simple utility to log an entire experiment run.
    
    Args:
        config: Configuration dictionary
        metrics: Final metrics dictionary
        experiment_name: MLflow experiment name
    """
    tracker = MLflowTracker(experiment_name)
    tracker.start_run(run_name=config.get("run_name", "default_run"))
    
    # Log all configurations
    tracker.log_params(config)
    
    # Log final metrics
    tracker.log_metrics(metrics)
    
    # Log config as artifact
    tracker.log_config(config)
    
    tracker.end_run()
    logger.info(f"Logged experiment: {experiment_name}")
