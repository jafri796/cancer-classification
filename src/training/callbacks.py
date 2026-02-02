"""
Formalized callback system for medical ML training orchestration.

Implements a production-grade callback interface that separates concerns:
- Model checkpointing and best-model tracking
- Early stopping with clinical thresholds
- Metrics logging and monitoring
- Experiment tracking integration (MLflow, WandB)
- Clinical validation hooks
- Model registry integration

Design Principles:
- Single responsibility: each callback handles one concern
- Composability: multiple callbacks work together
- Non-invasive: plugs into trainer without modifying core logic
- Medical compliance: all callbacks are deterministic and reproducible
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class Callback(ABC):
    """
    Base callback interface.
    
    All callbacks follow this pattern:
    - on_train_begin/end: Training session hooks
    - on_epoch_begin/end: Per-epoch hooks
    - on_train_epoch_end: After training epoch
    - on_val_epoch_end: After validation epoch
    """
    
    @abstractmethod
    def on_train_begin(self, trainer: 'Trainer', config: Dict[str, Any]) -> None:
        """Called at the start of training."""
        pass
    
    @abstractmethod
    def on_train_end(self, trainer: 'Trainer', history: Dict[str, Any]) -> None:
        """Called at the end of training."""
        pass
    
    @abstractmethod
    def on_epoch_begin(self, trainer: 'Trainer', epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    @abstractmethod
    def on_epoch_end(self, trainer: 'Trainer', epoch: int, logs: Dict[str, float]) -> None:
        """Called at the end of each epoch."""
        pass
    
    @abstractmethod
    def on_train_epoch_end(
        self,
        trainer: 'Trainer',
        epoch: int,
        train_metrics: Dict[str, float]
    ) -> None:
        """Called after training epoch finishes."""
        pass
    
    @abstractmethod
    def on_val_epoch_end(
        self,
        trainer: 'Trainer',
        epoch: int,
        val_metrics: Dict[str, float],
        clinical_metrics: Dict[str, Any]
    ) -> None:
        """Called after validation epoch finishes."""
        pass


class CallbackList:
    """Manages a list of callbacks."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def add(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)
    
    def on_train_begin(self, trainer, config):
        for callback in self.callbacks:
            callback.on_train_begin(trainer, config)
    
    def on_train_end(self, trainer, history):
        for callback in self.callbacks:
            callback.on_train_end(trainer, history)
    
    def on_epoch_begin(self, trainer, epoch):
        for callback in self.callbacks:
            callback.on_epoch_begin(trainer, epoch)
    
    def on_epoch_end(self, trainer, epoch, logs):
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, logs)
    
    def on_train_epoch_end(self, trainer, epoch, train_metrics):
        for callback in self.callbacks:
            callback.on_train_epoch_end(trainer, epoch, train_metrics)
    
    def on_val_epoch_end(self, trainer, epoch, val_metrics, clinical_metrics):
        for callback in self.callbacks:
            callback.on_val_epoch_end(trainer, epoch, val_metrics, clinical_metrics)


class ModelCheckpoint(Callback):
    """
    Save model checkpoints at intervals and when best validation metric is achieved.
    
    Medical Compliance:
    - Saves complete state for reproducibility audit
    - Tracks best model by clinical metric (AUC)
    - Saves every N epochs for failure recovery
    - Immutable checkpoint naming for regulatory compliance
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        save_frequency: Save checkpoint every N epochs
        monitor_metric: Metric to monitor for best model ('auc', 'loss', etc.)
        mode: 'min' or 'max' (minimize loss, maximize AUC)
        save_best_only: Only save checkpoints that improve best metric
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        save_frequency: int = 5,
        monitor_metric: str = 'auc',
        mode: str = 'max',
        save_best_only: bool = False,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_frequency = save_frequency
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.save_best_only = save_best_only
        
        # Track best metric
        self.best_metric = -float('inf') if mode == 'max' else float('inf')
        self.best_epoch = -1
        
        logger.info(
            f"ModelCheckpoint initialized: "
            f"dir={checkpoint_dir}, "
            f"monitor={monitor_metric}, "
            f"mode={mode}"
        )
    
    def on_train_begin(self, trainer, config):
        """Initialize checkpoint directory and log config."""
        logger.info(f"Checkpoints will be saved to {self.checkpoint_dir}")
    
    def on_train_end(self, trainer, history):
        """Log training completion."""
        logger.info(
            f"Training completed. "
            f"Best model at epoch {self.best_epoch} "
            f"with {self.monitor_metric}={self.best_metric:.4f}"
        )
    
    def on_epoch_begin(self, trainer, epoch):
        pass
    
    def on_epoch_end(self, trainer, epoch, logs):
        pass
    
    def on_train_epoch_end(self, trainer, epoch, train_metrics):
        pass
    
    def on_val_epoch_end(self, trainer, epoch, val_metrics, clinical_metrics):
        """
        Save checkpoint if condition met.
        
        Condition: every N epochs OR new best metric achieved
        """
        # Check if this is a best model
        current_metric = val_metrics.get(self.monitor_metric, float('inf'))
        is_best = self._is_improvement(current_metric)
        
        # Save if best or regular interval
        should_save_regular = (epoch + 1) % self.save_frequency == 0
        should_save_best = is_best and not self.save_best_only
        should_save = should_save_regular or (is_best and not self.save_best_only) or (is_best and self.save_best_only)
        
        if should_save:
            self._save_checkpoint(trainer, epoch, val_metrics, is_best=is_best)
            
            if is_best:
                self.best_metric = current_metric
                self.best_epoch = epoch
                logger.info(
                    f"Epoch {epoch}: New best model! "
                    f"{self.monitor_metric}={current_metric:.4f}"
                )
    
    def _is_improvement(self, current_metric: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'max':
            return current_metric > self.best_metric
        else:
            return current_metric < self.best_metric
    
    def _save_checkpoint(
        self,
        trainer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save checkpoint to disk."""
        checkpoint = {
            'epoch': epoch,
            'global_step': trainer.global_step,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
            'scaler_state_dict': trainer.scaler.state_dict() if trainer.scaler else None,
            'metrics': metrics,
            'config': trainer.config,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.debug(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")


class EarlyStopping(Callback):
    """
    Stop training when validation metric stops improving.
    
    Medical Compliance:
    - Prevents overfitting (critical for clinical generalization)
    - Configurable patience and improvement threshold
    - Explicit logging of early stopping trigger
    
    Args:
        monitor_metric: Metric to monitor ('auc', 'loss', etc.)
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum improvement to count as an improvement
        mode: 'min' or 'max'
        restore_best: Whether to restore best model weights
    """
    
    def __init__(
        self,
        monitor_metric: str = 'auc',
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = 'max',
        restore_best: bool = True,
    ):
        self.monitor_metric = monitor_metric
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        
        self.wait_count = 0
        self.best_metric = -float('inf') if mode == 'max' else float('inf')
        self.best_epoch = -1
        self.best_weights = None
        
        logger.info(
            f"EarlyStopping initialized: "
            f"monitor={monitor_metric}, "
            f"patience={patience}, "
            f"mode={mode}"
        )
    
    def on_train_begin(self, trainer, config):
        """Reset early stopping state."""
        self.wait_count = 0
        self.best_metric = -float('inf') if self.mode == 'max' else float('inf')
    
    def on_train_end(self, trainer, history):
        """Log early stopping result."""
        logger.info(
            f"Early stopping: best model at epoch {self.best_epoch} "
            f"with {self.monitor_metric}={self.best_metric:.4f}"
        )
    
    def on_epoch_begin(self, trainer, epoch):
        pass
    
    def on_epoch_end(self, trainer, epoch, logs):
        pass
    
    def on_train_epoch_end(self, trainer, epoch, train_metrics):
        pass
    
    def on_val_epoch_end(self, trainer, epoch, val_metrics, clinical_metrics):
        """Check if training should stop."""
        current_metric = val_metrics.get(self.monitor_metric, float('inf'))
        
        if self._is_improvement(current_metric):
            self.wait_count = 0
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.best_weights = {
                k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
            }
        else:
            self.wait_count += 1
            
            if self.wait_count >= self.patience:
                logger.warning(
                    f"Early stopping triggered after {self.wait_count} epochs "
                    f"with no improvement in {self.monitor_metric}"
                )
                trainer.stop_training = True
    
    def _is_improvement(self, current_metric: float) -> bool:
        """Check if current metric is better than best."""
        if self.mode == 'max':
            return current_metric > self.best_metric + self.min_delta
        else:
            return current_metric < self.best_metric - self.min_delta


class MetricsLogger(Callback):
    """
    Log metrics to external tracking systems.
    
    Integrates with MLflow for experiment tracking.
    
    Args:
        log_frequency: Log metrics every N batches (if supported)
    """
    
    def __init__(self, log_frequency: int = 100):
        self.log_frequency = log_frequency
    
    def on_train_begin(self, trainer, config):
        """Log training configuration."""
        if trainer.experiment_tracker:
            trainer.experiment_tracker.log_config(config)
            logger.info("Logged config to experiment tracker")
    
    def on_train_end(self, trainer, history):
        """Log final metrics summary."""
        if trainer.experiment_tracker:
            best_val_auc = max(m['auc'] for m in history['val'])
            trainer.experiment_tracker.log_metrics({
                'best_val_auc': best_val_auc,
                'total_epochs': len(history['val']),
            })
    
    def on_epoch_begin(self, trainer, epoch):
        pass
    
    def on_epoch_end(self, trainer, epoch, logs):
        pass
    
    def on_train_epoch_end(self, trainer, epoch, train_metrics):
        """Log training metrics."""
        if trainer.experiment_tracker:
            trainer.experiment_tracker.log_metrics(
                {f'train/{k}': v for k, v in train_metrics.items()},
                step=epoch
            )
    
    def on_val_epoch_end(self, trainer, epoch, val_metrics, clinical_metrics):
        """Log validation and clinical metrics."""
        if trainer.experiment_tracker:
            # Standard metrics
            trainer.experiment_tracker.log_metrics(
                {f'val/{k}': v for k, v in val_metrics.items()},
                step=epoch
            )
            
            # Clinical metrics
            if 'best_threshold' in clinical_metrics:
                best_thresh = clinical_metrics['best_threshold']
                best_metrics = clinical_metrics[f'threshold_{best_thresh}']
                trainer.experiment_tracker.log_metrics(
                    {
                        f'clinical/sensitivity': best_metrics['sensitivity'],
                        f'clinical/specificity': best_metrics['specificity'],
                        f'clinical/ppv': best_metrics['ppv'],
                        f'clinical/npv': best_metrics['npv'],
                    },
                    step=epoch
                )


class LearningRateMonitor(Callback):
    """Monitor and log learning rate changes."""
    
    def on_train_begin(self, trainer, config):
        pass
    
    def on_train_end(self, trainer, history):
        pass
    
    def on_epoch_begin(self, trainer, epoch):
        pass
    
    def on_epoch_end(self, trainer, epoch, logs):
        """Log current learning rate."""
        current_lr = trainer.optimizer.param_groups[0]['lr']
        logs['learning_rate'] = current_lr
        
        if epoch % 10 == 0:
            logger.debug(f"Epoch {epoch}: LR = {current_lr:.2e}")
    
    def on_train_epoch_end(self, trainer, epoch, train_metrics):
        pass
    
    def on_val_epoch_end(self, trainer, epoch, val_metrics, clinical_metrics):
        pass


class GradientMonitor(Callback):
    """Monitor gradient flow for debugging."""
    
    def on_train_begin(self, trainer, config):
        pass
    
    def on_train_end(self, trainer, history):
        pass
    
    def on_epoch_begin(self, trainer, epoch):
        pass
    
    def on_epoch_end(self, trainer, epoch, logs):
        pass
    
    def on_train_epoch_end(self, trainer, epoch, train_metrics):
        """Compute gradient statistics."""
        total_norm = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        grad_norm = total_norm ** 0.5
        train_metrics['grad_norm'] = grad_norm
        
        if epoch % 10 == 0 and grad_norm > 10.0:
            logger.warning(f"High gradient norm at epoch {epoch}: {grad_norm:.4f}")
    
    def on_val_epoch_end(self, trainer, epoch, val_metrics, clinical_metrics):
        pass


class ClinicalMetricsLogger(Callback):
    """
    Log and monitor clinical metrics (sensitivity, specificity, PPV, NPV).
    
    Alerts if metrics fall below clinical targets.
    """
    
    def __init__(
        self,
        target_sensitivity: float = 0.95,
        target_specificity: float = 0.90,
    ):
        self.target_sensitivity = target_sensitivity
        self.target_specificity = target_specificity
    
    def on_train_begin(self, trainer, config):
        logger.info(
            f"Clinical targets: "
            f"Sensitivity ≥ {self.target_sensitivity:.2%}, "
            f"Specificity ≥ {self.target_specificity:.2%}"
        )
    
    def on_train_end(self, trainer, history):
        pass
    
    def on_epoch_begin(self, trainer, epoch):
        pass
    
    def on_epoch_end(self, trainer, epoch, logs):
        pass
    
    def on_train_epoch_end(self, trainer, epoch, train_metrics):
        pass
    
    def on_val_epoch_end(self, trainer, epoch, val_metrics, clinical_metrics):
        """Log clinical metrics and alert on subthreshold performance."""
        if 'best_threshold' in clinical_metrics:
            best_thresh = clinical_metrics['best_threshold']
            best_metrics = clinical_metrics[f'threshold_{best_thresh}']
            
            sensitivity = best_metrics['sensitivity']
            specificity = best_metrics['specificity']
            
            logger.info(
                f"Epoch {epoch} Clinical Metrics "
                f"(threshold={best_thresh:.2f}): "
                f"Sens={sensitivity:.4f}, Spec={specificity:.4f}"
            )
            
            # Alert if below targets
            if sensitivity < self.target_sensitivity:
                logger.warning(
                    f"⚠️ Sensitivity {sensitivity:.4f} below target {self.target_sensitivity:.4f}"
                )
            
            if specificity < self.target_specificity:
                logger.warning(
                    f"⚠️ Specificity {specificity:.4f} below target {self.target_specificity:.4f}"
                )


class ModelRegistryCallback(Callback):
    """
    Register best model to model registry on training completion.
    
    Integration point with src/inference/model_registry.py
    """
    
    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path
        self.best_model_path = None
    
    def on_train_begin(self, trainer, config):
        pass
    
    def on_train_end(self, trainer, history):
        """Register best model to registry."""
        if self.best_model_path and self.registry_path:
            logger.info(f"Would register {self.best_model_path} to {self.registry_path}")
            # Integration with model_registry.py will happen here
    
    def on_epoch_begin(self, trainer, epoch):
        pass
    
    def on_epoch_end(self, trainer, epoch, logs):
        pass
    
    def on_train_epoch_end(self, trainer, epoch, train_metrics):
        pass
    
    def on_val_epoch_end(self, trainer, epoch, val_metrics, clinical_metrics):
        pass


def get_default_callbacks(
    checkpoint_dir: Path,
    config: Dict[str, Any],
    experiment_tracker: Optional[Any] = None,
) -> CallbackList:
    """
    Create default callback setup for production training.
    
    Includes:
    - Model checkpointing (best + regular)
    - Early stopping (clinical thresholds)
    - Metrics logging (MLflow)
    - Clinical metrics monitoring
    - Learning rate monitoring
    
    Args:
        checkpoint_dir: Directory for checkpoints
        config: Training configuration
        experiment_tracker: Optional MLflow/WandB tracker
        
    Returns:
        Configured CallbackList ready for use
    """
    callbacks = CallbackList()
    
    # Model checkpointing
    callbacks.add(
        ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            save_frequency=config.get('save_frequency', 5),
            monitor_metric='auc',
            mode='max',
        )
    )
    
    # Early stopping
    callbacks.add(
        EarlyStopping(
            monitor_metric='auc',
            patience=config.get('early_stopping_patience', 10),
            min_delta=0.0001,
            mode='max',
        )
    )
    
    # Metrics logging
    callbacks.add(MetricsLogger())
    
    # Clinical metrics
    callbacks.add(
        ClinicalMetricsLogger(
            target_sensitivity=0.95,
            target_specificity=0.90,
        )
    )
    
    # Learning rate monitor
    callbacks.add(LearningRateMonitor())
    
    # Gradient monitor
    callbacks.add(GradientMonitor())
    
    return callbacks
