"""
Production-grade training orchestrator for medical image classification.

Implements:
- Medical-grade metrics tracking (sensitivity, specificity, AUC)
- Early stopping with clinical thresholds
- Model checkpointing with best/latest savers
- Mixed precision training for speed
- Gradient accumulation for large effective batch sizes
- Comprehensive logging and experiment tracking
- MLflow integration for reproducibility
- Deterministic behavior across runs
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import logging
from tqdm import tqdm
import time
import numpy as np
from datetime import datetime

from .metrics import MedicalMetrics
from .losses import FocalLoss, WeightedBCELoss, AsymmetricLoss, ClinicalLoss
from .callbacks import CallbackList, get_default_callbacks
from ..utils.logging_utils import setup_logger, log_metrics
from ..utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


class Trainer:
    """
    Medical-grade training orchestrator for clinical validation workflows.
    
    Responsibilities:
    - Training loop with deterministic behavior
    - Validation and clinical metric tracking (sensitivity, specificity, AUC)
    - Checkpointing (best model + latest checkpoint)
    - Learning rate scheduling with adaptive decay
    - Mixed precision training (AMP) for efficiency
    - Gradient accumulation for large effective batch sizes
    - Early stopping with clinical thresholds
    - Experiment tracking with MLflow
    
    Clinical Requirements:
    - All random operations seeded for reproducibility
    - All configs logged for regulatory compliance
    - Metrics tracked at each epoch
    - Best model based on clinically meaningful metrics
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict
        device: Compute device (cuda/cpu)
        experiment_tracker: Optional MLflow/WandB tracker
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        experiment_tracker: Optional[Any] = None,
    ):
        # Ensure reproducibility
        seed = config.get('seed', 42)
        set_seed(seed, deterministic=True, benchmark=False)
        
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_tracker = experiment_tracker
        self.seed = seed
        
        # Log training configuration for compliance
        logger.info(f"Training config: {config}")
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = -float('inf')
        self.patience_counter = 0
        
        # Setup loss function
        self.criterion = self._setup_loss()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup mixed precision training
        self.use_amp = config.get('mixed_precision', {}).get('enabled', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup metrics tracker (clinical-grade)
        self.metrics = MedicalMetrics(
            threshold=config.get('threshold', 0.5),
            device=device
        )
        
        # Gradient accumulation for large effective batch sizes
        self.accumulation_steps = config.get('accumulation_steps', 1)
        
        # Checkpointing for best and latest models
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'experiments/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Log initialization
        logger.info(f"Trainer initialized on {device}")
        logger.info(f"Seed set to {seed} for reproducibility")
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping', {}).get('patience', 15)
        self.early_stopping_min_delta = config.get('early_stopping', {}).get('min_delta', 0.0001)
        
        # Callback system (formalized callbacks for monitoring, checkpointing, etc.)
        self.callbacks = get_default_callbacks(
            checkpoint_dir=self.checkpoint_dir,
            config=self.config,
            experiment_tracker=self.experiment_tracker
        )
        
        # Training control flag (can be set by callbacks like EarlyStopping)
        self.stop_training = False
        
        # Log training configuration for reproducibility
        logger.info(f"Training configuration: {self.config}")
        
        # Seed random operations for reproducibility
        set_seed(self.config.get('seed', 42))
        
        logger.info(f"Trainer initialized on {device}")
    
    def _setup_loss(self) -> nn.Module:
        """Setup loss function based on config."""
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'focal_loss')
        
        if loss_type == 'focal_loss':
            criterion = FocalLoss(
                alpha=loss_config.get('focal_alpha', 0.25),
                gamma=loss_config.get('focal_gamma', 2.0),
            )
            logger.info(f"Using Focal Loss (α={loss_config.get('focal_alpha')}, γ={loss_config.get('focal_gamma')})")
        
        elif loss_type == 'weighted_bce':
            pos_weight = torch.tensor([loss_config.get('pos_weight', 1.5)]).to(self.device)
            criterion = WeightedBCELoss(pos_weight=pos_weight)
            logger.info(f"Using Weighted BCE (pos_weight={loss_config.get('pos_weight')})")

        elif loss_type == 'asymmetric':
            criterion = AsymmetricLoss(
                gamma_pos=loss_config.get('gamma_pos', 0.0),
                gamma_neg=loss_config.get('gamma_neg', 4.0),
                clip=loss_config.get('clip', 0.05),
            )
            logger.info("Using Asymmetric Loss")

        elif loss_type == 'clinical':
            criterion = ClinicalLoss(
                fn_weight=loss_config.get('fn_weight', 3.0),
                focal_alpha=loss_config.get('focal_alpha', 0.25),
                focal_gamma=loss_config.get('focal_gamma', 2.0),
            )
            logger.info("Using Clinical Loss (FN-weighted)")
        
        else:
            criterion = nn.BCEWithLogitsLoss()
            logger.info("Using standard BCE Loss")
        
        return criterion
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with layer-wise learning rates."""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adamw')
        
        # Separate backbone and classifier parameters
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'classifier' in name or 'fc' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        # Different learning rates for backbone vs classifier
        base_lr = opt_config.get('lr', 1e-4)
        param_groups = [
            {'params': backbone_params, 'lr': base_lr},
            {'params': classifier_params, 'lr': base_lr * 10},  # Higher LR for classifier
        ]
        
        if opt_type == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=opt_config.get('betas', [0.9, 0.999]),
                weight_decay=opt_config.get('weight_decay', 1e-5),
                eps=opt_config.get('eps', 1e-8),
            )
        elif opt_type == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                betas=opt_config.get('betas', [0.9, 0.999]),
                weight_decay=opt_config.get('weight_decay', 1e-5),
            )
        elif opt_type == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 1e-5),
                nesterov=opt_config.get('nesterov', True),
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
        
        logger.info(f"Optimizer: {opt_type}, base_lr={base_lr}")
        return optimizer
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        sched_config = self.config.get('lr_scheduler', {})
        sched_type = sched_config.get('type', 'cosine_annealing_warm_restarts')
        
        if sched_type == 'cosine_annealing_warm_restarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_config.get('T_0', 10),
                T_mult=sched_config.get('T_mult', 2),
                eta_min=sched_config.get('eta_min', 1e-7),
            )
        elif sched_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=sched_config.get('eta_min', 1e-7),
            )
        elif sched_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=sched_config.get('reduce_factor', 0.5),
                patience=sched_config.get('reduce_patience', 5),
            )
        else:
            scheduler = None
        
        if scheduler:
            logger.info(f"LR Scheduler: {sched_type}")
        
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        self.metrics.reset()
        
        epoch_loss = 0.0
        batch_count = 0
        
        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            leave=False
        )
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.config.get('gradient_clipping', {}).get('enabled', True):
                    max_norm = self.config['gradient_clipping']['max_norm']
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update metrics
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                self.metrics.update(probs, labels)
            
            # Track loss
            epoch_loss += loss.item() * self.accumulation_steps
            batch_count += 1
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f"{epoch_loss / batch_count:.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
        
        # Compute epoch metrics
        metrics = self.metrics.compute()
        metrics['loss'] = epoch_loss / batch_count
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Validate model on validation set.
        
        Returns:
            metrics: Validation metrics
            clinical_metrics: Clinical validation metrics
        """
        self.model.eval()
        self.metrics.reset()
        
        val_loss = 0.0
        all_probs = []
        all_labels = []
        
        for images, labels in tqdm(self.val_loader, desc="Validation", leave=False):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            
            # Update metrics
            probs = torch.sigmoid(logits)
            self.metrics.update(probs, labels)
            
            # Store for clinical validation
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            val_loss += loss.item()
        
        # Compute standard metrics
        metrics = self.metrics.compute()
        metrics['loss'] = val_loss / len(self.val_loader)
        
        # Clinical validation at multiple thresholds
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        clinical_metrics = self._clinical_validation(all_probs, all_labels)
        
        return metrics, clinical_metrics
    
    def _clinical_validation(
        self,
        probs: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform clinical validation at multiple operating points.
        
        Medical requirements:
        - Sensitivity ≥ 95% (minimize false negatives)
        - Specificity ≥ 90% (minimize false positives)
        """
        thresholds = self.config.get('clinical_thresholds', [0.3, 0.4, 0.5, 0.6, 0.7])
        
        results = {}
        for threshold in thresholds:
            preds = (probs >= threshold).astype(int)
            
            # Compute confusion matrix elements
            tp = ((preds == 1) & (labels == 1)).sum()
            tn = ((preds == 0) & (labels == 0)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            
            # Clinical metrics
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            results[f'threshold_{threshold}'] = {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
            }
        
        # Find best threshold using Youden's index
        best_threshold = max(
            thresholds,
            key=lambda t: results[f'threshold_{t}']['sensitivity'] + 
                         results[f'threshold_{t}']['specificity']
        )
        results['best_threshold'] = best_threshold
        
        return results
    
    def train(self, epochs: int) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        history = {
            'train': [],
            'val': [],
            'clinical': [],
        }
        
        # Callback: training begin
        self.callbacks.on_train_begin(self, self.config)
        
        for epoch in range(epochs):
            if self.stop_training:
                logger.info("Stopping training as requested by callback")
                break
            
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Callback: epoch begin
            self.callbacks.on_epoch_begin(self, epoch)
            
            # Training
            train_metrics = self.train_epoch()
            history['train'].append(train_metrics)
            
            # Callback: training epoch end
            self.callbacks.on_train_epoch_end(self, epoch, train_metrics)
            
            # Validation
            val_metrics, clinical_metrics = self.validate()
            history['val'].append(val_metrics)
            history['clinical'].append(clinical_metrics)
            
            # Callback: validation epoch end
            self.callbacks.on_val_epoch_end(self, epoch, val_metrics, clinical_metrics)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['auc'])
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Logging
            logger.info(
                f"Epoch {epoch}/{epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train AUC: {train_metrics['auc']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}, "
                f"Val Sens: {val_metrics['sensitivity']:.4f}, "
                f"Val Spec: {val_metrics['specificity']:.4f}"
            )
            
            # Experiment tracking
            if self.experiment_tracker:
                self.experiment_tracker.log_metrics({
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()},
                    'epoch': epoch,
                }, step=epoch)
            
            # Callback: epoch end (after all metrics computed)
            epoch_logs = {**train_metrics, **val_metrics}
            self.callbacks.on_epoch_end(self, epoch, epoch_logs)
        
        logger.info("Training completed")
        
        # Callback: training end
        self.callbacks.on_train_end(self, history)
        
        return history
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'config': self.config,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")
        
        return checkpoint