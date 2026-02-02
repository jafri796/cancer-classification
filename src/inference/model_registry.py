"""
Pretrained model registry and loader with semantic audits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import logging
import torch
import yaml

from src.models.center_aware_resnet import create_center_aware_resnet50
from src.models.resnet_cbam import create_resnet50_cbam
from src.models.efficientnet import create_efficientnet
from src.models.vit import create_vit
from src.models.deit import create_deit_small

logger = logging.getLogger(__name__)


@dataclass
class PretrainedSpec:
    model_id: str
    architecture: str
    weights: str  # URL or local path
    input_size: Tuple[int, int]
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    label_semantics: str
    state_dict_key: Optional[str] = None
    strict: bool = True
    notes: Optional[str] = None


def _load_yaml(path: Union[str, Path]) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _resolve_weights(weights: str) -> str:
    if weights.startswith("http://") or weights.startswith("https://"):
        return weights
    return str(Path(weights).expanduser())


def _load_state_dict(weights: str) -> Dict[str, torch.Tensor]:
    resolved = _resolve_weights(weights)
    if resolved.startswith("http"):
        state = torch.hub.load_state_dict_from_url(resolved, map_location="cpu")
    else:
        state = torch.load(resolved, map_location="cpu")
    return state


def _build_model(architecture: str, model_cfg: Dict) -> torch.nn.Module:
    if architecture == "resnet50_cbam":
        return create_resnet50_cbam(model_cfg)
    if architecture.startswith("resnet"):
        return create_center_aware_resnet50(model_cfg)
    if architecture.startswith("efficientnet"):
        return create_efficientnet(model_cfg)
    if architecture == "deit_small":
        return create_deit_small(model_cfg)
    if architecture.startswith("vit"):
        return create_vit(model_cfg)
    raise ValueError(f"Unsupported architecture: {architecture}")


def load_pretrained_model(
    registry_path: Union[str, Path],
    model_id: str,
    model_cfg_path: Union[str, Path],
) -> Tuple[torch.nn.Module, PretrainedSpec]:
    registry = _load_yaml(registry_path)
    model_cfg = _load_yaml(model_cfg_path)

    entry = registry["pretrained_models"].get(model_id)
    if not entry:
        raise ValueError(f"Unknown pretrained model_id: {model_id}")

    spec = PretrainedSpec(
        model_id=model_id,
        architecture=entry["architecture"],
        weights=entry["weights"],
        input_size=tuple(entry["input_size"]),
        mean=tuple(entry["mean"]),
        std=tuple(entry["std"]),
        label_semantics=entry["label_semantics"],
        state_dict_key=entry.get("state_dict_key"),
        strict=entry.get("strict", True),
        notes=entry.get("notes"),
    )

    if spec.label_semantics != "pcam_center_32x32":
        raise ValueError(
            f"Pretrained model {model_id} has incompatible label semantics: {spec.label_semantics}"
        )
    if spec.input_size != (96, 96):
        raise ValueError(
            f"Pretrained model {model_id} expects {spec.input_size}, not 96x96"
        )

    cfg = model_cfg["models"].get(spec.architecture)
    if cfg is None:
        raise ValueError(f"Model config missing for architecture: {spec.architecture}")

    cfg = dict(cfg)
    if spec.weights.startswith("torchvision://") or spec.weights.startswith("timm://"):
        cfg["pretrained"] = True
        model = _build_model(spec.architecture, cfg)
    else:
        model = _build_model(spec.architecture, cfg)
        state = _load_state_dict(spec.weights)
        if spec.state_dict_key:
            state = state[spec.state_dict_key]
        model.load_state_dict(state, strict=spec.strict)
    model.eval()

    logger.info(f"Loaded pretrained model {model_id} ({spec.architecture})")
    return model, spec
