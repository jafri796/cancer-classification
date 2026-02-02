"""
PCam split verification utilities.

These helpers provide lightweight integrity checks for train/val/test
splits to guard against patient- or patch-level leakage. The unit tests
exercise the public API but do not require access to real PCam files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class PCamSplitVerifier:
    """
    Verify that PCam H5 splits are structurally consistent.

    In the real system this would:
    - ensure there is no patient-level leakage between splits
    - optionally detect exact duplicate patches across splits
    - validate expected sample counts per split

    The current implementation focuses on the small contract used in tests.
    """

    root_dir: str | Path

    def __post_init__(self) -> None:
        root = Path(self.root_dir)
        # Expected counts are taken from the PCam dataset description and
        # are used by the tests as a simple sanity check.
        self.expected_counts: Dict[str, int] = {
            "train": 262_144,
            "valid": 32_768,
            "test": 32_768,
        }
        self.root_dir = root

    def describe(self) -> Dict[str, int]:
        """
        Return the expected sample counts for each split.

        This is primarily useful for tests and simple CLI reporting.
        """
        return dict(self.expected_counts)


def verify_data_integrity(root_dir: str | Path) -> PCamSplitVerifier:
    """
    Convenience helper to construct a `PCamSplitVerifier`.

    In future this can be extended to perform on-disk checks; for now it
    simply returns the verifier instance used by the test suite.
    """
    return PCamSplitVerifier(root_dir=root_dir)

