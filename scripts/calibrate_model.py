"""
Calibrate pretrained model with temperature scaling and thresholds.
"""

import argparse
from pathlib import Path
import logging
import numpy as np
import torch
import yaml

from src.data.dataset import PCamDataset, create_dataloaders
from src.data.preprocessing import get_transforms
from src.inference.calibration import TemperatureScaler, optimize_thresholds, compute_ece
from src.inference.model_registry import load_pretrained_model
from src.models.center_aware_resnet import create_center_aware_resnet50
from src.models.efficientnet import create_efficientnet
from src.models.vit import create_vit


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_normalize_config(data_cfg):
    norm = data_cfg["preprocessing"]["normalization"]
    return {"normalize_mean": norm["mean"], "normalize_std": norm["std"]}


def _load_model_from_checkpoint(model_cfg, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = checkpoint.get("model_name")
    if model_name is None:
        raise ValueError("Checkpoint missing model_name")
    cfg = model_cfg["models"].get(model_name)
    if cfg is None and model_name.startswith("efficientnet_"):
        cfg = model_cfg["models"].get("efficientnet_b3", {}).copy()
        cfg["architecture"] = model_name.replace("_", "-")
    if cfg is None:
        raise ValueError(f"Model config not found for {model_name}")
    if model_name.startswith("resnet"):
        model = create_center_aware_resnet50(cfg)
    elif model_name.startswith("efficientnet"):
        model = create_efficientnet(cfg)
    elif model_name.startswith("vit"):
        model = create_vit(cfg)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_name


def main():
    parser = argparse.ArgumentParser(description="Calibrate PCam model")
    parser.add_argument("--data-config", type=str, default="config/data_config.yaml")
    parser.add_argument("--model-config", type=str, default="config/model_config.yaml")
    parser.add_argument("--registry", type=str, default="config/pretrained_registry.yaml")
    parser.add_argument("--pretrained-id", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="models/calibration.yaml")
    args = parser.parse_args()

    if not args.pretrained_id and not args.checkpoint:
        raise SystemExit("Provide --pretrained-id or --checkpoint")

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)

    data_dir = Path(data_cfg["dataset"]["data_dir"])
    norm_transform = get_transforms("val", _build_normalize_config(data_cfg))

    x_name, y_name = data_cfg["dataset"]["valid_x"], data_cfg["dataset"]["valid_y"]
    dataset = PCamDataset(
        x_path=str(data_dir / x_name),
        y_path=str(data_dir / y_name),
        transform=norm_transform,
        stain_normalizer=None,
        cache_normalized=False,
    )
    loader_cfg = {
        "train_batch_size": 1,
        "val_batch_size": data_cfg["dataloader"]["validation"]["batch_size"],
        "test_batch_size": data_cfg["dataloader"]["test"]["batch_size"],
        "num_workers": data_cfg["dataloader"]["validation"]["num_workers"],
        "pin_memory": data_cfg["dataloader"]["validation"]["pin_memory"],
    }
    _, eval_loader = create_dataloaders(
        config=loader_cfg, train_dataset=dataset, val_dataset=dataset
    )

    if args.pretrained_id:
        model, spec = load_pretrained_model(
            registry_path=args.registry,
            model_id=args.pretrained_id,
            model_cfg_path=args.model_config,
        )
        model_name = spec.model_id
    else:
        model, model_name = _load_model_from_checkpoint(model_cfg, Path(args.checkpoint))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_logits = []
    all_labels = []
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    scaler = TemperatureScaler()
    temperature = scaler.fit(
        all_logits, 
        all_labels,
        fit_set_name="validation",  # CRITICAL: Explicit validation-set declaration
    )
    scaled_logits = all_logits / max(temperature, 1e-6)
    probs = 1 / (1 + np.exp(-scaled_logits))

    raw_probs = 1 / (1 + np.exp(-all_logits))
    ece_before = compute_ece(raw_probs, all_labels)
    ece_after = compute_ece(probs, all_labels)

    thresholds = optimize_thresholds(
        probs,
        all_labels,
        target_sensitivity=0.95,
        target_specificity=0.90,
    )

    output = {
        "model_name": model_name,
        "temperature": temperature,
        "thresholds": thresholds,
        "ece_before": ece_before,
        "ece_after": ece_after,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.safe_dump(output, f)

    logger.info(f"Calibration saved to {args.output}")


if __name__ == "__main__":
    main()
