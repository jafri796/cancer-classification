"""
Reproducibility utilities for deterministic behavior across all libraries.
Critical for clinical-grade ML systems - ensures experiments are repeatable.
"""
import logging
import os
import random
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False) -> None:
    """
    Set all random seeds for reproducibility across libraries.

    Args:
        seed: Random seed value.
        deterministic: Force deterministic behavior in CUDA operations.
        benchmark: Enable cuDNN benchmarking (trades off determinism for speed).

    Medical Rationale:
    - Reproducibility is non-negotiable for clinical validation
    - Same input + same seed = identical output across runs
    - Required for regulatory approval and peer review
    """
    # Python random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # PyTorch random
    torch.manual_seed(seed)

    # CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Environment variables for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Configure cuBLAS workspace for determinism when required
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    # Deterministic behavior
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark) and not deterministic

    logger.info(
        "Random seeds set (seed=%s, deterministic=%s, benchmark=%s)",
        seed,
        deterministic,
        benchmark,
    )


def get_seed_state() -> Dict[str, Any]:
    """Get current seed state for logging/debugging."""
    return {
        "python_seed": random.getstate(),
        "numpy_seed": np.random.get_state()[1][0],
        "torch_seed": torch.initial_seed(),
    }


class ReproducibilityContext:
    """Context manager for reproducible code blocks."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.saved_state: Dict[str, Any] | None = None

    def __enter__(self) -> "ReproducibilityContext":
        """Save current RNG state and set a temporary seed."""
        self.saved_state = {
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        set_seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Restore previous RNG state."""
        if self.saved_state is not None:
            random.setstate(self.saved_state["random"])
            np.random.set_state(self.saved_state["numpy"])
            torch.set_rng_state(self.saved_state["torch"])
        return False


# Module-level seed setting for import-time reproducibility
_SEED = int(os.environ.get("PCAM_SEED", "42"))
set_seed(_SEED, deterministic=True, benchmark=False)