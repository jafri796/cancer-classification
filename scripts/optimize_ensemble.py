"""
Optimize ensemble weights for PCam center-region detection.

Pipeline:
1. Load multiple trained model checkpoints
2. Run each model on the VALIDATION set to collect logits
3. Calibrate each model via temperature scaling (validation set only)
4. Optimize ensemble weights (maximize AUC on validation set)
5. Output ensemble configuration YAML

CRITICAL: All fitting (calibration + weight optimization) uses the
VALIDATION set exclusively.  The test set must remain held-out.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import PCamDataset, create_dataloaders
from src.data.preprocessing import get_transforms
from src.inference.calibration import (
    TemperatureScaler,
    compute_ece,
    optimize_ensemble_weights,
    optimize_thresholds,
)
from src.inference.model_registry import _build_model
from src.utils.reproducibility import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_normalize_config(data_cfg: Dict) -> Dict:
    norm = data_cfg["preprocessing"]["normalization"]
    return {"normalize_mean": norm["mean"], "normalize_std": norm["std"]}


def _load_checkpoint(
    checkpoint_path: str,
    model_cfg: Dict,
    device: torch.device,
) -> Tuple[torch.nn.Module, str]:
    """Load a model from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Resolve model name stored in the checkpoint
    model_name = ckpt.get("model_name") or ckpt.get("config", {}).get("model_name")
    if model_name is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} missing 'model_name'. "
            "Ensure the training script stores it in the checkpoint dict."
        )

    # Build architecture config
    cfg = model_cfg.get("models", {}).get(model_name)
    if cfg is None and model_name.startswith("efficientnet_"):
        cfg = model_cfg.get("models", {}).get("efficientnet_b3", {}).copy()
        cfg["architecture"] = model_name.replace("_", "-")
    if cfg is None:
        raise ValueError(f"Model config not found for {model_name}")

    cfg = dict(cfg)
    model = _build_model(model_name, cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    logger.info(f"Loaded checkpoint {checkpoint_path} (arch={model_name})")
    return model, model_name


def _collect_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on a dataloader and return (logits, labels) as numpy arrays."""
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_logits).flatten(), np.concatenate(all_labels).flatten()


def _calibrate_model(
    logits: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Fit temperature scaling and return (temperature, calibrated_probs)."""
    scaler = TemperatureScaler()
    temperature = scaler.fit(logits, labels, fit_set_name="validation")
    scaled_logits = logits / max(temperature, 1e-6)
    probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    return temperature, probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optimize ensemble weights on the validation set."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Paths to model checkpoint files (.pt)",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="config/data_config.yaml",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="config/model_config.yaml",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--metric",
        choices=["auc", "ece"],
        default="auc",
        help="Metric to optimize ensemble weights for (default: auc)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=5000,
        help="Number of random trials for weight search",
    )
    parser.add_argument(
        "--target-sensitivity",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--target-specificity",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="config/ensemble_config.yaml",
        help="Output path for ensemble configuration YAML",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    set_seed(args.seed, deterministic=True, benchmark=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load configs ----
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)

    # ---- Build validation dataloader ----
    data_dir = Path(data_cfg["dataset"]["data_dir"])
    norm_cfg = _build_normalize_config(data_cfg)
    val_transform = get_transforms("val", norm_cfg)

    val_dataset = PCamDataset(
        x_path=str(data_dir / data_cfg["dataset"]["valid_x"]),
        y_path=str(data_dir / data_cfg["dataset"]["valid_y"]),
        transform=val_transform,
        stain_normalizer=None,
        cache_normalized=False,
    )
    loader_cfg = {
        "train_batch_size": 1,
        "val_batch_size": data_cfg["dataloader"]["validation"]["batch_size"],
        "test_batch_size": data_cfg["dataloader"]["validation"]["batch_size"],
        "num_workers": data_cfg["dataloader"]["validation"]["num_workers"],
        "pin_memory": data_cfg["dataloader"]["validation"]["pin_memory"],
    }
    _, val_loader = create_dataloaders(
        config=loader_cfg, train_dataset=val_dataset, val_dataset=val_dataset
    )

    logger.info(f"Validation set: {len(val_dataset)} samples")

    # ---- Collect per-model logits & calibrate ----
    model_names: List[str] = []
    temperatures: List[float] = []
    calibrated_probs: List[np.ndarray] = []
    labels: np.ndarray = None

    for ckpt_path in args.checkpoints:
        model, name = _load_checkpoint(ckpt_path, model_cfg, device)
        logits, lbls = _collect_logits(model, val_loader, device)

        # Verify labels are consistent across models
        if labels is None:
            labels = lbls
        else:
            assert np.array_equal(labels, lbls), "Label mismatch across dataloader runs"

        temperature, probs = _calibrate_model(logits, labels)

        raw_probs = 1.0 / (1.0 + np.exp(-logits))
        ece_before = compute_ece(raw_probs, labels)
        ece_after = compute_ece(probs, labels)
        logger.info(
            f"  {name}: temperature={temperature:.4f}, "
            f"ECE {ece_before:.4f} -> {ece_after:.4f}"
        )

        model_names.append(name)
        temperatures.append(temperature)
        calibrated_probs.append(probs)

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ---- Optimize ensemble weights ----
    logger.info(f"Optimizing ensemble weights (metric={args.metric}, trials={args.n_trials})")
    weights = optimize_ensemble_weights(
        model_probs=calibrated_probs,
        labels=labels,
        metric=args.metric,
        n_trials=args.n_trials,
    )

    # ---- Compute ensemble-level calibration metrics ----
    ensemble_probs = np.zeros_like(labels, dtype=np.float64)
    for w, p in zip(weights, calibrated_probs):
        ensemble_probs += w * p
    ensemble_ece = compute_ece(ensemble_probs, labels)

    # ---- Optimize ensemble thresholds ----
    thresholds = optimize_thresholds(
        ensemble_probs,
        labels,
        target_sensitivity=args.target_sensitivity,
        target_specificity=args.target_specificity,
    )

    logger.info(f"Ensemble ECE: {ensemble_ece:.4f}")
    logger.info(f"Ensemble thresholds: {thresholds}")

    # ---- Build output config ----
    ensemble_config = {
        "_description": (
            "Ensemble configuration. Weights and calibration fitted on "
            "VALIDATION set only. Do NOT re-fit on test set."
        ),
        "seed": args.seed,
        "optimization_metric": args.metric,
        "n_trials": args.n_trials,
        "ensemble_ece": float(ensemble_ece),
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "models": [],
    }
    for i, ckpt_path in enumerate(args.checkpoints):
        ensemble_config["models"].append({
            "checkpoint": str(Path(ckpt_path).resolve()),
            "model_name": model_names[i],
            "temperature": float(temperatures[i]),
            "weight": float(weights[i]),
        })

    # ---- Save ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(ensemble_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Ensemble config saved to {output_path}")

    # Also print a summary
    print("\n" + "=" * 60)
    print("ENSEMBLE OPTIMIZATION RESULTS")
    print("=" * 60)
    for i, name in enumerate(model_names):
        print(f"  {name:25s}  weight={weights[i]:.4f}  temp={temperatures[i]:.4f}")
    print(f"\n  Ensemble ECE:        {ensemble_ece:.4f}")
    print(f"  Balanced threshold:  {thresholds.get('balanced', 'N/A')}")
    print(f"  Sensitivity thresh:  {thresholds.get('sensitivity', 'N/A')}")
    print(f"  Specificity thresh:  {thresholds.get('specificity', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
