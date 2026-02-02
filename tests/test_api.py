"""
API tests for FastAPI inference service.
"""

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from fastapi.testclient import TestClient


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 96 * 96, 1),
        )

    def forward(self, x):
        return self.net(x)


def test_api_predict_endpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        model_path = tmpdir / "dummy_model.pt"

        # Save dummy model
        model = DummyModel().eval()
        torch.save(model, model_path)

        os.environ["MODEL_PATH"] = str(model_path)
        os.environ["MODEL_CONFIG"] = "config/model_config.yaml"
        os.environ["DATA_CONFIG"] = "config/data_config.yaml"

        from deployment.api.app import app

        client = TestClient(app)

        img = Image.fromarray(np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = client.post("/predict", files={"file": ("patch.png", buf, "image/png")})
        assert response.status_code == 200
        data = response.json()
        assert "probability" in data
        assert "label" in data
