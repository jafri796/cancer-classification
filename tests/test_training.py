"""
Unit tests for training pipeline components.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import Trainer


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3 * 96 * 96, 1),
        )

    def forward(self, x):
        return self.net(x)


def test_trainer_single_epoch_runs():
    device = torch.device("cpu")
    model = DummyModel()

    # Dummy dataset
    x = torch.randn(8, 3, 96, 96)
    y = torch.randint(0, 2, (8, 1)).float()
    dataset = TensorDataset(x, y)

    train_loader = DataLoader(dataset, batch_size=4)
    val_loader = DataLoader(dataset, batch_size=4)

    config = {
        "loss": {"type": "bce"},
        "optimizer": {"type": "adamw", "lr": 1e-3},
        "lr_scheduler": {"type": "cosine"},
        "mixed_precision": {"enabled": False},
        "accumulation_steps": 1,
        "gradient_clipping": {"enabled": False},
        "clinical_thresholds": [0.5],
        "checkpoint_dir": "experiments/test_checkpoints",
        "save_frequency": 1,
        "early_stopping": {"patience": 1, "min_delta": 0.0},
        "epochs": 1,
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    history = trainer.train(epochs=1)
    assert len(history["train"]) == 1
    assert len(history["val"]) == 1
