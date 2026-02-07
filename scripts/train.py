#!/usr/bin/env python
"""
Train PCam classification models.

Main entry point for training center-aware models on the PCam dataset.
Supports multiple architectures, mixed precision training, and MLflow tracking.

Usage:
    python scripts/train.py --config config/training_config.yaml
    python scripts/train.py --config config/training_config.yaml --model resnet50_se
    python scripts/train.py --config config/training_config.yaml --resume experiments/run_001/checkpoints/last.pt
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import create_dataloaders
from src.data.preprocessing import StainNormalizer, get_transforms
from src.models import (
    create_center_aware_resnet50,
    create_resnet50_cbam,
    create_efficientnet,
    create_vit,
    create_deit_small,
)
from src.training.trainer import Trainer
from src.training.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    MetricsLogger,
    LearningRateMonitor,
)
from src.utils.reproducibility import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_FACTORIES = {
    "resnet50_se": create_center_aware_resnet50,
    "resnet50_cbam": create_resnet50_cbam,
    "efficientnet_b0": lambda **kw: create_efficientnet(variant="b0", **kw),
    "efficientnet_b1": lambda **kw: create_efficientnet(variant="b1", **kw),
    "efficientnet_b2": lambda **kw: create_efficientnet(variant="b2", **kw),
    "vit_b16": create_vit,
    "deit_small": create_deit_small,
}


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_experiment(config: dict, model_name: str) -> Path:
    """Create experiment directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{model_name}_{timestamp}"
    exp_dir = Path(config.get("output_dir", "experiments")) / exp_name
    
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "plots").mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Experiment directory: {exp_dir}")
    return exp_dir


def create_model(model_name: str, config: dict) -> torch.nn.Module:
    """Create model from configuration."""
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_FACTORIES.keys())}")
    
    model_config = config.get("model", {})
    factory = MODEL_FACTORIES[model_name]
    
    model = factory(
        num_classes=model_config.get("num_classes", 1),
        pretrained=model_config.get("pretrained", True),
        center_size=model_config.get("center_size", 32),
        dropout_rate=model_config.get("dropout_rate", 0.3),
    )
    
    logger.info(f"Created model: {model_name}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def create_callbacks(exp_dir: Path, config: dict) -> list:
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpoint
    callbacks.append(ModelCheckpoint(
        checkpoint_dir=exp_dir / "checkpoints",
        save_frequency=5,
        monitor_metric="auc",
        mode="max",
        save_best_only=False,
    ))
    
    # Early stopping
    early_stop_config = config.get("early_stopping", {})
    if early_stop_config.get("enabled", True):
        callbacks.append(EarlyStopping(
            monitor_metric="auc",
            mode="max",
            patience=early_stop_config.get("patience", 15),
            min_delta=early_stop_config.get("min_delta", 0.0001),
        ))
    
    # Metrics logger
    callbacks.append(MetricsLogger(
        log_frequency=100,
    ))
    
    # Learning rate monitor
    callbacks.append(LearningRateMonitor())
    
    return callbacks


def main():
    parser = argparse.ArgumentParser(
        description="Train PCam classification model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODEL_FACTORIES.keys()),
        help="Model architecture to train (overrides config)"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: cuda if available)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Set seed for reproducibility
    seed = args.seed or config.get("seed", 42)
    set_seed(seed)
    logger.info(f"Random seed: {seed}")
    
    # Determine model
    model_name = args.model or config.get("model", {}).get("architecture", "resnet50_se")
    logger.info(f"Training model: {model_name}")
    
    # Setup experiment
    exp_dir = setup_experiment(config, model_name)
    
    # Create datasets and data loaders
    data_config = config.get("data", {})
    data_dir = Path(data_config.get("data_dir", "data/raw"))
    batch_size = data_config.get("batch_size", 64)
    num_workers = data_config.get("num_workers", 4)

    from src.data.dataset import PCamDataset
    train_transform = get_transforms('train', data_config)
    val_transform = get_transforms('val', data_config)

    train_dataset = PCamDataset(
        x_path=str(data_dir / "camelyonpatch_level_2_split_train_x.h5"),
        y_path=str(data_dir / "camelyonpatch_level_2_split_train_y.h5"),
        transform=train_transform,
    )
    val_dataset = PCamDataset(
        x_path=str(data_dir / "camelyonpatch_level_2_split_valid_x.h5"),
        y_path=str(data_dir / "camelyonpatch_level_2_split_valid_y.h5"),
        transform=val_transform,
    )

    loader_config = {
        'train_batch_size': batch_size,
        'val_batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
    }
    train_loader, val_loader = create_dataloaders(
        config=loader_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    logger.info(f"Train samples: {len(train_loader.dataset):,}")
    logger.info(f"Val samples: {len(val_loader.dataset):,}")
    
    # Create model
    model = create_model(model_name, config)
    model = model.to(args.device)
    
    # Create callbacks
    callbacks = create_callbacks(exp_dir, config)
    
    # Create trainer
    training_config = config.get("training", {})
    training_config['checkpoint_dir'] = str(exp_dir / "checkpoints")
    device = torch.device(args.device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
    )
    
    # Resume if specified
    if args.resume:
        if not args.resume.exists():
            logger.error(f"Checkpoint not found: {args.resume}")
            sys.exit(1)
        trainer.load_checkpoint(str(args.resume))
        logger.info(f"Resumed from: {args.resume}")
    
    # Train (use two-phase if configured, otherwise single-phase)
    logger.info("Starting training...")
    epochs = training_config.get("epochs", 50)
    use_two_phase = training_config.get("two_phase", {}).get("enabled", True)
    try:
        if use_two_phase:
            phase1 = training_config.get("two_phase", {}).get("phase1_epochs", 5)
            phase2 = training_config.get("two_phase", {}).get("phase2_epochs", epochs - phase1)
            history = trainer.train_two_phase(
                phase1_epochs=phase1,
                phase2_epochs=phase2,
            )
        else:
            history = trainer.train(epochs=epochs)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint(
            epoch=trainer.current_epoch,
            metrics={},
            is_best=False,
        )
    
    logger.info(f"\n✓ Training complete. Results saved to: {exp_dir}")


if __name__ == "__main__":
    main()
