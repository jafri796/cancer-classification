"""
Inference tests for predictor utilities.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.inference.predictor import PCamPredictor


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 96 * 96, 1),
        )

    def forward(self, x):
        return self.net(x)


def test_predictor_runs_on_dummy_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        model_path = tmpdir / "dummy_model.pt"
        img_path = tmpdir / "patch.png"

        # Save dummy model
        model = DummyModel().eval()
        torch.save(model, model_path)

        # Create dummy 96x96 patch
        img = Image.fromarray(np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8))
        img.save(img_path)

        predictor = PCamPredictor(
            model_path=str(model_path),
            model_config_path="config/model_config.yaml",
            data_config_path="config/data_config.yaml",
            device="cpu",
        )

        result = predictor.predict(str(img_path))
        assert "probability" in result
        assert "label" in result
