"""
Shared pytest fixtures for PCam classification tests.

This module provides reusable fixtures for:
- Model instances
- Mock datasets and dataloaders
- Device configuration
- Temporary directories
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image


@pytest.fixture(scope="session")
def device():
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def cpu_device():
    """Force CPU device for deterministic tests."""
    return torch.device("cpu")


@pytest.fixture
def sample_image():
    """Create a sample 96x96 RGB image tensor."""
    return torch.randn(1, 3, 96, 96)


@pytest.fixture
def sample_batch():
    """Create a sample batch of images and labels."""
    images = torch.randn(8, 3, 96, 96)
    labels = torch.randint(0, 2, (8,)).float()
    return images, labels


@pytest.fixture
def sample_numpy_image():
    """Create a sample 96x96 RGB numpy image."""
    return np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8)


@pytest.fixture
def sample_pil_image(sample_numpy_image):
    """Create a sample PIL Image."""
    return Image.fromarray(sample_numpy_image)


@pytest.fixture
def mock_h5_data(tmp_path):
    """Create mock H5 data files for testing."""
    import h5py
    
    n_samples = 100
    
    # Create train files
    train_x_path = tmp_path / "camelyonpatch_level_2_split_train_x.h5"
    train_y_path = tmp_path / "camelyonpatch_level_2_split_train_y.h5"
    
    with h5py.File(train_x_path, "w") as f:
        f.create_dataset("x", data=np.random.randint(0, 256, (n_samples, 96, 96, 3), dtype=np.uint8))
    
    with h5py.File(train_y_path, "w") as f:
        f.create_dataset("y", data=np.random.randint(0, 2, (n_samples, 1, 1, 1), dtype=np.uint8))
    
    # Create valid files
    valid_x_path = tmp_path / "camelyonpatch_level_2_split_valid_x.h5"
    valid_y_path = tmp_path / "camelyonpatch_level_2_split_valid_y.h5"
    
    with h5py.File(valid_x_path, "w") as f:
        f.create_dataset("x", data=np.random.randint(0, 256, (n_samples // 4, 96, 96, 3), dtype=np.uint8))
    
    with h5py.File(valid_y_path, "w") as f:
        f.create_dataset("y", data=np.random.randint(0, 2, (n_samples // 4, 1, 1, 1), dtype=np.uint8))
    
    # Create test files
    test_x_path = tmp_path / "camelyonpatch_level_2_split_test_x.h5"
    test_y_path = tmp_path / "camelyonpatch_level_2_split_test_y.h5"
    
    with h5py.File(test_x_path, "w") as f:
        f.create_dataset("x", data=np.random.randint(0, 256, (n_samples // 4, 96, 96, 3), dtype=np.uint8))
    
    with h5py.File(test_y_path, "w") as f:
        f.create_dataset("y", data=np.random.randint(0, 2, (n_samples // 4, 1, 1, 1), dtype=np.uint8))
    
    return tmp_path


@pytest.fixture
def mock_stain_reference(tmp_path):
    """Create a mock stain reference image."""
    ref_path = tmp_path / "reference_patch.png"
    img = Image.fromarray(np.random.randint(0, 256, (96, 96, 3), dtype=np.uint8))
    img.save(ref_path)
    return ref_path


@pytest.fixture
def resnet_model():
    """Create a ResNet model for testing."""
    from src.models import create_center_aware_resnet50
    model = create_center_aware_resnet50(
        num_classes=1,
        pretrained=False,
        center_size=32,
    )
    return model


@pytest.fixture
def efficientnet_model():
    """Create an EfficientNet model for testing."""
    from src.models import create_efficientnet
    model = create_efficientnet(
        variant="b0",
        num_classes=1,
        pretrained=False,
        center_size=32,
    )
    return model


@pytest.fixture
def vit_model():
    """Create a ViT model for testing."""
    from src.models import create_vit
    model = create_vit(
        num_classes=1,
        pretrained=False,
        center_size=32,
    )
    return model


@pytest.fixture
def trained_model_checkpoint(tmp_path, resnet_model):
    """Create a mock trained model checkpoint."""
    checkpoint_path = tmp_path / "model.pt"
    torch.save({
        "model_state_dict": resnet_model.state_dict(),
        "model_class": "CenterAwareResNet50SE",
        "model_config": {
            "num_classes": 1,
            "pretrained": False,
            "center_size": 32,
        },
        "epoch": 10,
        "val_auc": 0.95,
    }, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def mock_config():
    """Create a mock configuration dictionary."""
    return {
        "data": {
            "data_dir": "data/raw",
            "batch_size": 32,
            "num_workers": 0,
        },
        "model": {
            "architecture": "resnet50_se",
            "num_classes": 1,
            "pretrained": False,
            "center_size": 32,
            "dropout_rate": 0.3,
        },
        "training": {
            "epochs": 10,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "mixed_precision": False,
            "gradient_accumulation": 1,
        },
        "early_stopping": {
            "enabled": True,
            "patience": 5,
            "min_delta": 0.001,
        },
        "seed": 42,
        "output_dir": "experiments",
    }


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility in all tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def disable_gpu():
    """Temporarily disable GPU for testing."""
    original_cuda_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    yield
    torch.cuda.is_available = original_cuda_available
