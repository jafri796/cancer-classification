"""
Evaluate a trained PCam model on validation or test split.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from src.data.dataset import PCamDataset, create_dataloaders
from src.data.preprocessing import get_transforms, StainNormalizer
from src.training.metrics import MedicalMetrics
from src.models.center_aware_resnet import create_center_aware_resnet50
from src.models.resnet_cbam import create_resnet50_cbam
from src.models.efficientnet import create_efficientnet
from src.models.vit import create_vit
from src.models.deit import create_deit_small
from src.inference.model_registry import load_pretrained_model
from src.inference.ensemble_predictor import EnsemblePredictor
from src.inference.calibration import compute_ece


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _build_train_aug_config(data_cfg):
    aug = data_cfg["augmentation"]["train"]
    return {
        "rotation_angles": aug["random_rotation"]["angles"],
        "horizontal_flip_prob": aug["random_horizontal_flip"]["probability"],
        "vertical_flip_prob": aug["random_vertical_flip"]["probability"],
        "color_jitter_brightness": aug["color_jitter"]["brightness"],
        "color_jitter_contrast": aug["color_jitter"]["contrast"],
        "color_jitter_saturation": aug["color_jitter"]["saturation"],
        "color_jitter_hue": aug["color_jitter"]["hue"],
        "color_jitter_prob": aug["color_jitter"]["probability"],
    }


def _build_normalize_config(data_cfg):
    norm = data_cfg["preprocessing"]["normalization"]
    return {"normalize_mean": norm["mean"], "normalize_std": norm["std"]}


def _load_stain_normalizer(data_cfg):
    stain_cfg = data_cfg["preprocessing"]["stain_normalization"]
    if not stain_cfg.get("enabled", False):
        return None
    ref_path = Path(stain_cfg.get("reference_image", ""))
    if not ref_path.exists():
        raise FileNotFoundError(f"Stain reference missing: {ref_path}")
    normalizer = StainNormalizer(
        luminosity_threshold=stain_cfg["luminosity_threshold"],
        angular_percentile=stain_cfg["angular_percentile"],
    )
    from PIL import Image

    ref_img = np.array(Image.open(ref_path))
    normalizer.fit(ref_img)
    return normalizer


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

    if model_name.startswith("resnet50_cbam"):
        model = create_resnet50_cbam(cfg)
    elif model_name.startswith("resnet"):
        model = create_center_aware_resnet50(cfg)
    elif model_name.startswith("efficientnet"):
        model = create_efficientnet(cfg)
    elif model_name.startswith("deit"):
        model = create_deit_small(cfg)
    elif model_name.startswith("vit"):
        model = create_vit(cfg)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, model_name


def main():
    parser = argparse.ArgumentParser(description="Evaluate PCam model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--registry", type=str, default="config/pretrained_registry.yaml")
    parser.add_argument("--pretrained-id", type=str, default=None)
    parser.add_argument("--deployment-config", type=str, default="config/deployment_config.yaml")
    parser.add_argument("--ensemble", action="store_true", help="Evaluate ensemble from deployment config")
    parser.add_argument("--data-config", type=str, default="config/data_config.yaml")
    parser.add_argument("--model-config", type=str, default="config/model_config.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if not args.ensemble and not args.checkpoint and not args.pretrained_id:
        raise SystemExit("Provide --checkpoint, --pretrained-id, or --ensemble")

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)

    data_dir = Path(data_cfg["dataset"]["data_dir"])
    norm_transform = get_transforms("val", _build_normalize_config(data_cfg))

    if args.split == "val":
        x_name, y_name = data_cfg["dataset"]["valid_x"], data_cfg["dataset"]["valid_y"]
    else:
        x_name, y_name = data_cfg["dataset"]["test_x"], data_cfg["dataset"]["test_y"]

    stain_normalizer = _load_stain_normalizer(data_cfg)
    dataset = PCamDataset(
        x_path=str(data_dir / x_name),
        y_path=str(data_dir / y_name),
        transform=None if args.ensemble else norm_transform,
        stain_normalizer=stain_normalizer,
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

    if args.ensemble:
        with open(args.deployment_config, "r") as f:
            deploy_cfg = yaml.safe_load(f)
        model = None
        model_name = "ensemble"
        ensemble = EnsemblePredictor(
            models=deploy_cfg["ensemble"]["members"],
            model_config_path=deploy_cfg["model"]["model_config"],
            data_config_path=deploy_cfg["model"]["data_config"],
            threshold=float(deploy_cfg["api"].get("threshold", 0.5)),
            use_tta=bool(deploy_cfg["api"].get("use_tta", False)),
            device=args.device,
        )
    elif args.pretrained_id:
        model, spec = load_pretrained_model(args.registry, args.pretrained_id, args.model_config)
        model_name = spec.model_id
    else:
        model, model_name = _load_model_from_checkpoint(model_cfg, Path(args.checkpoint))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if model is not None:
        model = model.to(device)

    metrics = MedicalMetrics(device=device)
    metrics.reset()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in eval_loader:
            if args.ensemble:
                batch_probs = []
                for i in range(images.size(0)):
                    img = images[i].cpu()
                    img = (img.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    pil = Image.fromarray(img)
                    result = ensemble.predict(pil)
                    batch_probs.append(result["probability"])
                probs = torch.tensor(batch_probs).view(-1, 1)
            else:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits)
            metrics.update(probs, labels)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    results = metrics.compute()
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    results["ece"] = compute_ece(all_probs, all_labels)

    logger.info(f"Evaluation results for {model_name} ({args.split})")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.6f}")

    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
