#!/usr/bin/env python
"""
Verify PCam dataset integrity and structure.

Performs comprehensive validation of the downloaded PCam dataset:
- File existence and readability
- H5 structure validation
- Shape verification (96x96x3 patches)
- Label distribution analysis
- Data split consistency checks

Usage:
    python scripts/verify_data.py --data-dir data/raw
    python scripts/verify_data.py --data-dir data/raw --detailed
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

EXPECTED_SPLITS = {
    "train": {"x": (262144, 96, 96, 3), "y": (262144, 1, 1, 1)},
    "valid": {"x": (32768, 96, 96, 3), "y": (32768, 1, 1, 1)},
    "test": {"x": (32768, 96, 96, 3), "y": (32768, 1, 1, 1)},
}


def check_file_exists(data_dir: Path, split: str, data_type: str) -> Tuple[bool, Path]:
    """Check if a specific data file exists."""
    filename = f"camelyonpatch_level_2_split_{split}_{data_type}.h5"
    filepath = data_dir / filename
    return filepath.exists(), filepath


def validate_h5_structure(filepath: Path, expected_key: str) -> Tuple[bool, str]:
    """Validate H5 file structure."""
    try:
        with h5py.File(filepath, "r") as f:
            if expected_key not in f.keys():
                return False, f"Missing key '{expected_key}' in {filepath.name}"
            return True, "OK"
    except Exception as e:
        return False, f"Cannot read {filepath.name}: {e}"


def validate_shape(filepath: Path, key: str, expected_shape: Tuple) -> Tuple[bool, str]:
    """Validate data shape."""
    try:
        with h5py.File(filepath, "r") as f:
            actual_shape = f[key].shape
            if actual_shape != expected_shape:
                return False, f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
            return True, f"Shape: {actual_shape}"
    except Exception as e:
        return False, f"Cannot read shape: {e}"


def analyze_labels(filepath: Path) -> Dict:
    """Analyze label distribution."""
    try:
        with h5py.File(filepath, "r") as f:
            labels = f["y"][:].flatten()
            unique, counts = np.unique(labels, return_counts=True)
            total = len(labels)
            distribution = {
                int(label): {
                    "count": int(count),
                    "percentage": float(count / total * 100)
                }
                for label, count in zip(unique, counts)
            }
            return {
                "total": total,
                "distribution": distribution,
                "balanced": abs(counts[0] - counts[1]) / total < 0.1 if len(counts) == 2 else False
            }
    except Exception as e:
        return {"error": str(e)}


def validate_pixel_range(filepath: Path, sample_size: int = 1000) -> Tuple[bool, str]:
    """Validate pixel value range (should be 0-255 for uint8)."""
    try:
        with h5py.File(filepath, "r") as f:
            data = f["x"]
            indices = np.random.choice(len(data), min(sample_size, len(data)), replace=False)
            samples = data[indices]
            
            min_val, max_val = samples.min(), samples.max()
            dtype = samples.dtype
            
            if dtype != np.uint8:
                return False, f"Unexpected dtype: {dtype} (expected uint8)"
            if min_val < 0 or max_val > 255:
                return False, f"Pixel range [{min_val}, {max_val}] outside [0, 255]"
            
            return True, f"dtype={dtype}, range=[{min_val}, {max_val}]"
    except Exception as e:
        return False, f"Cannot validate pixels: {e}"


def run_verification(data_dir: Path, detailed: bool = False) -> bool:
    """Run complete dataset verification."""
    data_dir = Path(data_dir)
    all_passed = True
    results = []
    
    logger.info(f"Verifying PCam dataset in: {data_dir}")
    logger.info("=" * 60)
    
    for split, expected in EXPECTED_SPLITS.items():
        logger.info(f"\n[{split.upper()}]")
        
        for data_type in ["x", "y"]:
            exists, filepath = check_file_exists(data_dir, split, data_type)
            if not exists:
                logger.error(f"  ✗ {filepath.name} - MISSING")
                all_passed = False
                continue
            
            logger.info(f"  ✓ {filepath.name} - EXISTS")
            
            # Validate H5 structure
            valid, msg = validate_h5_structure(filepath, data_type)
            if not valid:
                logger.error(f"    ✗ Structure: {msg}")
                all_passed = False
                continue
            
            # Validate shape
            valid, msg = validate_shape(filepath, data_type, expected[data_type])
            if not valid:
                logger.error(f"    ✗ Shape: {msg}")
                all_passed = False
            else:
                logger.info(f"    ✓ {msg}")
            
            # Additional checks for image data
            if data_type == "x" and detailed:
                valid, msg = validate_pixel_range(filepath)
                if not valid:
                    logger.warning(f"    ⚠ Pixel range: {msg}")
                else:
                    logger.info(f"    ✓ Pixels: {msg}")
            
            # Label analysis
            if data_type == "y":
                analysis = analyze_labels(filepath)
                if "error" not in analysis:
                    dist = analysis["distribution"]
                    bal_status = "balanced" if analysis["balanced"] else "imbalanced"
                    logger.info(f"    ✓ Labels: {analysis['total']} samples ({bal_status})")
                    if detailed:
                        for label, info in dist.items():
                            logger.info(f"      Class {label}: {info['count']} ({info['percentage']:.1f}%)")
    
    logger.info("\n" + "=" * 60)
    
    # Summary
    total_samples = sum(
        EXPECTED_SPLITS[split]["x"][0] for split in EXPECTED_SPLITS
    )
    logger.info(f"Total expected samples: {total_samples:,}")
    logger.info(f"  Train: {EXPECTED_SPLITS['train']['x'][0]:,}")
    logger.info(f"  Valid: {EXPECTED_SPLITS['valid']['x'][0]:,}")
    logger.info(f"  Test:  {EXPECTED_SPLITS['test']['x'][0]:,}")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify PCam dataset integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing PCam data files (default: data/raw)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Run detailed analysis including pixel range and label distribution"
    )
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        logger.error(f"Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    if run_verification(args.data_dir, detailed=args.detailed):
        logger.info("\n✓ VERIFICATION PASSED")
        sys.exit(0)
    else:
        logger.error("\n✗ VERIFICATION FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
