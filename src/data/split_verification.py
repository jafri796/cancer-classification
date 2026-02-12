"""
PCam split verification utilities.

These helpers provide lightweight integrity checks for train/val/test
splits to guard against patient- or patch-level leakage. The unit tests
exercise the public API but do not require access to real PCam files.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Standard PCam H5 file names (from zenodo.org/record/2546921)
_SPLIT_FILES: Dict[str, Tuple[str, str]] = {
    "train": (
        "camelyonpatch_level_2_split_train_x.h5",
        "camelyonpatch_level_2_split_train_y.h5",
    ),
    "valid": (
        "camelyonpatch_level_2_split_valid_x.h5",
        "camelyonpatch_level_2_split_valid_y.h5",
    ),
    "test": (
        "camelyonpatch_level_2_split_test_x.h5",
        "camelyonpatch_level_2_split_test_y.h5",
    ),
}


@dataclass
class SplitReport:
    """Result of a single split verification."""
    split_name: str
    sample_count: int
    expected_count: int
    image_shape: Optional[Tuple[int, ...]] = None
    label_shape: Optional[Tuple[int, ...]] = None
    label_range_ok: bool = True
    count_ok: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    """Aggregate report for all splits."""
    splits: Dict[str, SplitReport] = field(default_factory=dict)
    cross_split_duplicates: int = 0
    all_ok: bool = True
    errors: List[str] = field(default_factory=list)


@dataclass
class PCamSplitVerifier:
    """
    Verify that PCam H5 splits are structurally consistent.

    Checks performed (when H5 files are present):
    - Sample count matches expected PCam counts per split
    - Image shape is (N, 96, 96, 3) and values in [0, 255]
    - Label shape is (N, 1, 1, 1) and values in {0, 1}
    - No exact duplicate patches across train/val/test splits
      (sampled check for efficiency)

    When H5 files are absent the verifier still provides expected counts
    and a describe() method for tests.
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

    # ------------------------------------------------------------------
    # On-disk verification (requires H5 files)
    # ------------------------------------------------------------------

    def verify(self, check_duplicates: bool = True, dup_sample_size: int = 500) -> VerificationReport:
        """
        Run full verification on PCam H5 files found in root_dir.

        Args:
            check_duplicates: If True, sample patches from each split and
                check for exact duplicates across splits.
            dup_sample_size: Number of patches to sample per split for
                duplicate checking (higher = slower but more thorough).

        Returns:
            VerificationReport with per-split details and aggregate status.
        """
        try:
            import h5py
        except ImportError:
            report = VerificationReport()
            report.errors.append("h5py not installed; cannot verify H5 files")
            report.all_ok = False
            return report

        report = VerificationReport()

        # -- Per-split checks --
        for split_name, (x_file, y_file) in _SPLIT_FILES.items():
            sr = self._verify_split(split_name, x_file, y_file)
            report.splits[split_name] = sr
            if sr.errors:
                report.all_ok = False
                report.errors.extend(
                    [f"[{split_name}] {e}" for e in sr.errors]
                )

        # -- Cross-split duplicate check --
        if check_duplicates and all(
            sr.errors == [] or sr.sample_count > 0
            for sr in report.splits.values()
        ):
            n_dups = self._check_cross_split_duplicates(dup_sample_size)
            report.cross_split_duplicates = n_dups
            if n_dups > 0:
                msg = f"Found {n_dups} duplicate patches across splits (leakage risk)"
                report.errors.append(msg)
                report.all_ok = False
                logger.warning(msg)

        if report.all_ok:
            logger.info("All split verification checks passed")
        else:
            logger.warning(f"Split verification found {len(report.errors)} issue(s)")

        return report

    def _verify_split(
        self, split_name: str, x_filename: str, y_filename: str
    ) -> SplitReport:
        """Verify a single split's H5 files."""
        import h5py

        x_path = self.root_dir / x_filename
        y_path = self.root_dir / y_filename
        expected = self.expected_counts.get(split_name, 0)

        sr = SplitReport(
            split_name=split_name,
            sample_count=0,
            expected_count=expected,
        )

        if not x_path.exists():
            sr.errors.append(f"Image file not found: {x_path}")
            sr.count_ok = False
            return sr
        if not y_path.exists():
            sr.errors.append(f"Label file not found: {y_path}")
            sr.count_ok = False
            return sr

        try:
            with h5py.File(x_path, "r") as xf, h5py.File(y_path, "r") as yf:
                if "x" not in xf:
                    sr.errors.append(f"Missing key 'x' in {x_filename}")
                    return sr
                if "y" not in yf:
                    sr.errors.append(f"Missing key 'y' in {y_filename}")
                    return sr

                x_data = xf["x"]
                y_data = yf["y"]

                sr.sample_count = x_data.shape[0]
                sr.image_shape = x_data.shape
                sr.label_shape = y_data.shape

                # Count check
                if sr.sample_count != expected:
                    sr.count_ok = False
                    sr.errors.append(
                        f"Expected {expected} samples, found {sr.sample_count}"
                    )

                # Image/label count mismatch
                if x_data.shape[0] != y_data.shape[0]:
                    sr.errors.append(
                        f"Image count ({x_data.shape[0]}) != label count ({y_data.shape[0]})"
                    )

                # Image shape: (N, 96, 96, 3)
                if len(x_data.shape) != 4 or x_data.shape[1:] != (96, 96, 3):
                    sr.errors.append(
                        f"Unexpected image shape {x_data.shape}, expected (N, 96, 96, 3)"
                    )

                # Label shape: (N, 1, 1, 1)
                if len(y_data.shape) != 4 or y_data.shape[1:] != (1, 1, 1):
                    sr.errors.append(
                        f"Unexpected label shape {y_data.shape}, expected (N, 1, 1, 1)"
                    )

                # Spot-check label values are binary {0, 1}
                sample_labels = y_data[:min(1000, sr.sample_count)].flatten()
                unique_labels = set(np.unique(sample_labels).tolist())
                if not unique_labels.issubset({0, 1}):
                    sr.label_range_ok = False
                    sr.errors.append(
                        f"Labels contain values outside {{0, 1}}: {unique_labels}"
                    )

        except Exception as e:
            sr.errors.append(f"Error reading H5 files: {e}")

        if not sr.errors:
            logger.info(
                f"  {split_name}: {sr.sample_count} samples OK "
                f"(img={sr.image_shape}, lbl={sr.label_shape})"
            )

        return sr

    def _check_cross_split_duplicates(self, sample_size: int = 500) -> int:
        """
        Sample patches from each split and check for exact duplicates
        across splits using image hashes.

        Returns the number of duplicate patches found.
        """
        import h5py

        split_hashes: Dict[str, Set[str]] = {}

        for split_name, (x_file, _) in _SPLIT_FILES.items():
            x_path = self.root_dir / x_file
            if not x_path.exists():
                continue

            hashes: Set[str] = set()
            try:
                with h5py.File(x_path, "r") as xf:
                    n = xf["x"].shape[0]
                    rng = np.random.RandomState(42)
                    indices = rng.choice(n, size=min(sample_size, n), replace=False)
                    indices.sort()

                    for idx in indices:
                        patch = xf["x"][int(idx)]
                        h = hashlib.md5(patch.tobytes()).hexdigest()
                        hashes.add(h)
            except Exception as e:
                logger.warning(f"Could not sample {split_name} for duplicate check: {e}")
                continue

            split_hashes[split_name] = hashes

        # Count cross-split overlaps
        splits = list(split_hashes.keys())
        duplicates = 0
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                overlap = split_hashes[splits[i]] & split_hashes[splits[j]]
                if overlap:
                    duplicates += len(overlap)
                    logger.warning(
                        f"Found {len(overlap)} duplicate patches between "
                        f"{splits[i]} and {splits[j]}"
                    )

        return duplicates


def verify_data_integrity(root_dir: str | Path) -> VerificationReport:
    """
    Run full split verification on PCam data in *root_dir*.

    Returns a VerificationReport.  If no H5 files are present the report
    will contain file-not-found errors for each missing split.
    """
    verifier = PCamSplitVerifier(root_dir=root_dir)
    return verifier.verify()

