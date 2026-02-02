#!/usr/bin/env python
"""
Prepare stain normalization reference image.

Selects a representative histopathology patch to serve as the reference
for Macenko stain normalization. The reference should have good staining
quality and typical H&E characteristics.

Usage:
    python scripts/prepare_stain_reference.py --data-dir data/raw --output data/references/reference_patch.png
    python scripts/prepare_stain_reference.py --data-dir data/raw --output data/references/reference_patch.png --interactive
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_stain_metrics(image: np.ndarray) -> dict:
    """
    Compute metrics for stain quality assessment.
    
    Good reference images should have:
    - Adequate tissue content (not too much background)
    - Good color saturation
    - Typical H&E staining pattern
    """
    # Convert to float for calculations
    img_float = image.astype(np.float32) / 255.0
    
    # Tissue mask (non-white pixels)
    gray = np.mean(img_float, axis=2)
    tissue_mask = gray < 0.85
    tissue_ratio = np.mean(tissue_mask)
    
    # Color statistics in tissue regions
    if tissue_ratio > 0.1:
        tissue_pixels = img_float[tissue_mask]
        mean_rgb = tissue_pixels.mean(axis=0)
        std_rgb = tissue_pixels.std(axis=0)
        
        # H&E typically has pink/purple tones
        # Good reference: R > G, B moderate
        color_score = mean_rgb[0] - mean_rgb[1]  # R - G difference
        saturation = std_rgb.mean()
    else:
        color_score = 0
        saturation = 0
    
    return {
        "tissue_ratio": float(tissue_ratio),
        "color_score": float(color_score),
        "saturation": float(saturation),
        "composite_score": float(tissue_ratio * 0.4 + color_score * 0.3 + saturation * 0.3)
    }


def find_best_reference(
    data_path: Path,
    n_candidates: int = 1000,
    seed: int = 42
) -> Tuple[int, np.ndarray, dict]:
    """
    Find the best reference image from training data.
    
    Samples candidate images and selects the one with the best
    stain quality metrics.
    """
    np.random.seed(seed)
    
    with h5py.File(data_path, "r") as f:
        total_images = f["x"].shape[0]
        indices = np.random.choice(total_images, min(n_candidates, total_images), replace=False)
        
        best_idx = -1
        best_score = -float("inf")
        best_image = None
        best_metrics = None
        
        logger.info(f"Evaluating {len(indices)} candidate images...")
        
        for i, idx in enumerate(indices):
            image = f["x"][idx]
            metrics = compute_stain_metrics(image)
            
            # Filter: need adequate tissue content
            if metrics["tissue_ratio"] < 0.3:
                continue
            
            if metrics["composite_score"] > best_score:
                best_score = metrics["composite_score"]
                best_idx = int(idx)
                best_image = image.copy()
                best_metrics = metrics
            
            if (i + 1) % 200 == 0:
                logger.info(f"  Processed {i + 1}/{len(indices)} candidates...")
    
    return best_idx, best_image, best_metrics


def save_reference(
    image: np.ndarray,
    output_path: Path,
    metrics: dict,
    index: int
) -> None:
    """Save reference image and metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save image
    Image.fromarray(image).save(output_path)
    logger.info(f"Saved reference image to: {output_path}")
    
    # Save metadata
    metadata_path = output_path.with_suffix(".json")
    import json
    metadata = {
        "source_index": index,
        "metrics": metrics,
        "image_shape": list(image.shape),
        "description": "Stain normalization reference for Macenko method"
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to: {metadata_path}")


def interactive_selection(
    data_path: Path,
    n_options: int = 5,
    seed: int = 42
) -> Tuple[int, np.ndarray]:
    """
    Interactive reference selection mode.
    
    Shows top candidates and lets user choose.
    """
    np.random.seed(seed)
    
    with h5py.File(data_path, "r") as f:
        total_images = f["x"].shape[0]
        indices = np.random.choice(total_images, min(500, total_images), replace=False)
        
        candidates = []
        for idx in indices:
            image = f["x"][idx]
            metrics = compute_stain_metrics(image)
            if metrics["tissue_ratio"] >= 0.3:
                candidates.append((int(idx), image.copy(), metrics))
        
        # Sort by composite score
        candidates.sort(key=lambda x: x[2]["composite_score"], reverse=True)
        top_candidates = candidates[:n_options]
    
    logger.info(f"\nTop {len(top_candidates)} candidates:")
    for i, (idx, img, metrics) in enumerate(top_candidates):
        logger.info(f"  {i + 1}. Index {idx}: tissue={metrics['tissue_ratio']:.2f}, "
                   f"color={metrics['color_score']:.2f}, score={metrics['composite_score']:.3f}")
    
    # For non-interactive environments, just return the best
    logger.info("\nSelecting best candidate automatically...")
    return top_candidates[0][0], top_candidates[0][1]


def main():
    parser = argparse.ArgumentParser(
        description="Prepare stain normalization reference image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing PCam data (default: data/raw)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/references/reference_patch.png"),
        help="Output path for reference image (default: data/references/reference_patch.png)"
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=1000,
        help="Number of candidate images to evaluate (default: 1000)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive selection mode (shows top candidates)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Find training data
    train_x_path = args.data_dir / "camelyonpatch_level_2_split_train_x.h5"
    if not train_x_path.exists():
        logger.error(f"Training data not found: {train_x_path}")
        logger.error("Run 'python scripts/download_pcam.py' first")
        sys.exit(1)
    
    logger.info("Finding optimal stain reference image...")
    
    if args.interactive:
        idx, image = interactive_selection(train_x_path, seed=args.seed)
        metrics = compute_stain_metrics(image)
    else:
        idx, image, metrics = find_best_reference(
            train_x_path,
            n_candidates=args.n_candidates,
            seed=args.seed
        )
    
    if image is None:
        logger.error("Failed to find suitable reference image")
        sys.exit(1)
    
    logger.info(f"\nSelected reference: index {idx}")
    logger.info(f"  Tissue ratio: {metrics['tissue_ratio']:.2f}")
    logger.info(f"  Color score: {metrics['color_score']:.3f}")
    logger.info(f"  Composite score: {metrics['composite_score']:.3f}")
    
    save_reference(image, args.output, metrics, idx)
    logger.info("\n✓ Stain reference preparation complete")


if __name__ == "__main__":
    main()
