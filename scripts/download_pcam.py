#!/usr/bin/env python
"""
Download PCam (PatchCamelyon) dataset.

Downloads the official PCam dataset from the GigaScience repository.
The dataset consists of 96x96 pixel histopathology patches extracted
from Camelyon16 whole-slide images.

Usage:
    python scripts/download_pcam.py --data-dir data/raw
    python scripts/download_pcam.py --data-dir data/raw --verify-only
"""

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Official PCam dataset URLs (GigaScience GigaDB)
PCAM_FILES = {
    "camelyonpatch_level_2_split_train_x.h5": {
        "url": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz",
        "md5": "1571f514728f59571f2125eb07ae0e46",
        "size_gb": 6.2,
    },
    "camelyonpatch_level_2_split_train_y.h5": {
        "url": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz",
        "md5": "35c2d7259d906cfc8143347bb8e05be4",
        "size_gb": 0.001,
    },
    "camelyonpatch_level_2_split_valid_x.h5": {
        "url": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz",
        "md5": "d8c2d60d490dbd479f8199a6a5b0e0c5",
        "size_gb": 0.8,
    },
    "camelyonpatch_level_2_split_valid_y.h5": {
        "url": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz",
        "md5": "2fd5bfcd6d1d8e86e1a94b1ddee359be",
        "size_gb": 0.001,
    },
    "camelyonpatch_level_2_split_test_x.h5": {
        "url": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz",
        "md5": "7b754f2f32fffa96b4c5e35e8a0d8a8e",
        "size_gb": 0.8,
    },
    "camelyonpatch_level_2_split_test_y.h5": {
        "url": "https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz",
        "md5": "d4c19e290e05cbb3a1a5c35b9c0c9ab1",
        "size_gb": 0.001,
    },
}


class DownloadProgressBar(tqdm):
    """Progress bar for urlretrieve."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def compute_md5(filepath: Path, chunk_size: int = 8192) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_file(url: str, dest: Path, expected_md5: str = None) -> bool:
    """Download a file with progress bar and optional MD5 verification."""
    try:
        with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
            urlretrieve(url, dest, reporthook=t.update_to)
        
        if expected_md5:
            actual_md5 = compute_md5(dest)
            if actual_md5 != expected_md5:
                logger.error(f"MD5 mismatch for {dest.name}: expected {expected_md5}, got {actual_md5}")
                return False
            logger.info(f"MD5 verified for {dest.name}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def decompress_gzip(gz_path: Path, output_path: Path) -> bool:
    """Decompress a gzip file."""
    import gzip
    import shutil
    
    try:
        with gzip.open(gz_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception as e:
        logger.error(f"Failed to decompress {gz_path}: {e}")
        return False


def verify_dataset(data_dir: Path) -> bool:
    """Verify all dataset files exist and have correct structure."""
    import h5py
    
    all_valid = True
    for filename in PCAM_FILES.keys():
        filepath = data_dir / filename
        if not filepath.exists():
            logger.error(f"Missing file: {filepath}")
            all_valid = False
            continue
        
        try:
            with h5py.File(filepath, "r") as f:
                if "x" in filename:
                    assert "x" in f.keys(), f"Missing 'x' key in {filename}"
                    shape = f["x"].shape
                    logger.info(f"{filename}: shape {shape}")
                elif "y" in filename:
                    assert "y" in f.keys(), f"Missing 'y' key in {filename}"
                    shape = f["y"].shape
                    logger.info(f"{filename}: shape {shape}")
        except Exception as e:
            logger.error(f"Invalid H5 file {filepath}: {e}")
            all_valid = False
    
    return all_valid


def download_pcam(data_dir: Path, skip_existing: bool = True) -> bool:
    """Download the complete PCam dataset."""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    total_size = sum(info["size_gb"] for info in PCAM_FILES.values())
    logger.info(f"Total download size: ~{total_size:.1f} GB")
    
    success = True
    for filename, info in PCAM_FILES.items():
        output_path = data_dir / filename
        gz_path = data_dir / f"{filename}.gz"
        
        if skip_existing and output_path.exists():
            logger.info(f"Skipping {filename} (already exists)")
            continue
        
        logger.info(f"Downloading {filename}...")
        if not download_file(info["url"], gz_path):
            success = False
            continue
        
        logger.info(f"Decompressing {filename}...")
        if not decompress_gzip(gz_path, output_path):
            success = False
            continue
        
        # Clean up compressed file
        gz_path.unlink()
        logger.info(f"Completed {filename}")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Download PCam dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/download_pcam.py --data-dir data/raw
    python scripts/download_pcam.py --data-dir data/raw --verify-only
    python scripts/download_pcam.py --data-dir data/raw --force
        """
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to download data to (default: data/raw)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing files, don't download"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        logger.info(f"Verifying dataset in {args.data_dir}...")
        if verify_dataset(args.data_dir):
            logger.info("Dataset verification PASSED")
            sys.exit(0)
        else:
            logger.error("Dataset verification FAILED")
            sys.exit(1)
    
    logger.info(f"Downloading PCam dataset to {args.data_dir}...")
    if download_pcam(args.data_dir, skip_existing=not args.force):
        logger.info("Download complete!")
        if verify_dataset(args.data_dir):
            logger.info("Dataset verification PASSED")
            sys.exit(0)
        else:
            logger.error("Dataset verification FAILED")
            sys.exit(1)
    else:
        logger.error("Download failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
