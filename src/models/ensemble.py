"""
Weighted ensemble for PCam center-region detection.
"""

from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    """
    Weighted ensemble of binary classifiers.

    Outputs a logit derived from the weighted mean probability.
    """

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        if len(models) == 0:
            raise ValueError("Ensemble requires at least one model")
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        if len(weights) != len(models):
            raise ValueError("Weights length must match number of models")
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", weight_tensor / weight_tensor.sum())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = []
        for model in self.models:
            logits = model(x)
            probs.append(torch.sigmoid(logits))
        stacked = torch.stack(probs, dim=0)  # (M, B, 1)
        weighted = (stacked * self.weights.view(-1, 1, 1)).sum(dim=0)
        eps = 1e-6
        weighted = torch.clamp(weighted, eps, 1 - eps)
        logits = torch.log(weighted / (1 - weighted))
        return logits


class StackingEnsembleModel(nn.Module):
    """
    Implements a stacking ensemble model for combining predictions.

    Args:
        base_models (List[nn.Module]): List of base models.
        meta_model (nn.Module): Meta-model for combining base predictions.
    """

    def __init__(self, base_models: List[nn.Module], meta_model: nn.Module):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.meta_model = meta_model

    def forward(self, x):
        base_outputs = [model(x) for model in self.base_models]
        stacked_outputs = torch.stack(base_outputs, dim=1)  # Shape: (batch, num_models, ...)
        meta_input = stacked_outputs.mean(dim=1)  # Example: mean aggregation
        return self.meta_model(meta_input)