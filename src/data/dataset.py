"""
PCam Dataset Implementation
Production-grade PyTorch Dataset for histopathology image classification

Clinical Requirements:
- Validates data integrity (shapes, ranges, labels)
- Implements stain normalization for scanner invariance
- Supports medical-grade augmentation
- Handles edge cases gracefully
- Logs all operations for compliance
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable, Tuple, Dict, List
import logging
from PIL import Image

from .preprocessing import StainNormalizer, get_transforms, FitSetOrigin
from ..utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


class PCamDataset(Dataset):
    """
    PatchCamelyon Dataset for center-region tumor detection.
    
    Medical Rationale:
    - 96×96 RGB patches (standard PCam format)
    - Label: 1 if ≥1 tumor pixel in center 32×32, else 0
    - Lazy loading with h5py for memory efficiency
    - Stain normalization for scanner-invariant features
    - Medical-grade augmentation (center-preserving)
    
    Args:
        x_path: Path to H5 file containing images
        y_path: Path to H5 file containing labels
        transform: Augmentation/preprocessing transforms
        stain_normalizer: Stain normalization instance
        cache_normalized: Cache stain-normalized images
        validate_data: Perform data integrity checks
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        x_path: str,
        y_path: str,
        transform: Optional[Callable] = None,
        stain_normalizer: Optional[StainNormalizer] = None,
        cache_normalized: bool = False,
        validate_data: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        
        # Set seed for reproducibility in validation
        set_seed(seed)
        
        self.x_path = Path(x_path)
        self.y_path = Path(y_path)
        
        # Validate paths
        if not self.x_path.exists():
            raise FileNotFoundError(f"Image file not found: {self.x_path}")
        if not self.y_path.exists():
            raise FileNotFoundError(f"Label file not found: {self.y_path}")
        
        # CRITICAL: Validate stain normalizer fit-set origin
        if stain_normalizer is not None:
            if not stain_normalizer.is_fitted:
                raise ValueError(
                    "LEAKAGE RISK: Stain normalizer is not fitted. "
                    "Must fit normalizer on training set before creating datasets. "
                    "Call normalizer.fit(reference_image, declared_fit_set=FitSetOrigin.TRAINING_ONLY)"
                )
            
            # Validate fit-set origin is declared
            if stain_normalizer.fit_set_origin == FitSetOrigin.UNDECLARED:
                raise ValueError(
                    "LEAKAGE RISK: Stain normalizer fit-set origin is UNDECLARED. "
                    "Refitted normalizer with explicit fit-set declaration. "
                    "Example: normalizer.fit(image, declared_fit_set=FitSetOrigin.TRAINING_ONLY)"
                )
            
            logger.info(
                f"Stain normalizer validated (fit-set origin: {stain_normalizer.fit_set_origin.value})"
            )
        
        # Open H5 files (keep open for lazy loading)
        self.x_file = h5py.File(self.x_path, 'r')
        self.y_file = h5py.File(self.y_path, 'r')
        
        # Access datasets
        self.x_data = self.x_file['x']  # Shape: (N, 96, 96, 3)
        self.y_data = self.y_file['y']  # Shape: (N, 1, 1, 1)
        
        # Validate dimensions
        assert self.x_data.shape[0] == self.y_data.shape[0], \
            f"Mismatch: {self.x_data.shape[0]} images, {self.y_data.shape[0]} labels"
        
        self.n_samples = self.x_data.shape[0]
        logger.info(f"Loaded {self.n_samples} samples from {self.x_path.name}")
        
        # Store components
        self.transform = transform
        self.stain_normalizer = stain_normalizer
        self.cache_normalized = cache_normalized
        
        # Cache for normalized images (if enabled)
        self._cache: Dict[int, np.ndarray] = {}
        
        # Validate data integrity
        if validate_data:
            self._validate_dataset()
        
        # Compute dataset statistics
        self.class_counts = self._compute_class_distribution()
        self.positive_ratio = self.class_counts[1] / self.n_samples
        
        logger.info(
            f"Dataset statistics: "
            f"Positive: {self.class_counts[1]} ({self.positive_ratio:.2%}), "
            f"Negative: {self.class_counts[0]}"
        )
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample with preprocessing.
        
        Pipeline:
        1. Load image from H5
        2. Apply stain normalization (if configured)
        3. Apply augmentation transforms
        4. Convert to tensor
        5. Extract label
        
        Returns:
            image: Tensor of shape (3, 96, 96)
            label: Tensor of shape (1,) with value 0 or 1
        """
        try:
            # Check cache first
            if self.cache_normalized and idx in self._cache:
                image = self._cache[idx]
            else:
                # Load image from H5 (uint8, shape: (96, 96, 3))
                image = self.x_data[idx]
                
                # Validate image
                if image.shape != (96, 96, 3):
                    raise ValueError(f"Invalid image shape at index {idx}: {image.shape}")
                
                # Apply stain normalization
                if self.stain_normalizer is not None:
                    image = self.stain_normalizer.normalize(image)
                
                # Cache if enabled
                if self.cache_normalized:
                    self._cache[idx] = image.copy()
            
            # Convert to PIL Image for transforms
            image = Image.fromarray(image.astype(np.uint8))
            
            # Apply augmentation/preprocessing transforms
            if self.transform is not None:
                image = self.transform(image)
            else:
                # Default: Convert to tensor and normalize
                image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            
            # Load label (shape: (1, 1, 1) -> scalar)
            # PCam label semantics (non-negotiable):
            # 1 = tumor present in center 32x32 region
            # 0 = no tumor in center region (periphery irrelevant)
            label = float(self.y_data[idx, 0, 0, 0])
            label = torch.tensor([label], dtype=torch.float32)
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            raise
    
    def _validate_dataset(self) -> None:
        """
        Perform comprehensive data integrity checks.
        
        Clinical Requirements:
        - Images must be 96×96×3 RGB
        - Labels must be binary (0 or 1)
        - No NaN or Inf values
        - Consistent dimensions across dataset
        """
        logger.info("Validating dataset integrity...")
        
        # Check random samples with fixed seed for reproducibility
        n_samples_to_check = min(100, self.n_samples)
        indices = np.random.choice(self.n_samples, n_samples_to_check, replace=False)
        
        issues = []
        
        for idx in indices:
            try:
                # Check image
                img = self.x_data[idx]
                if img.shape != (96, 96, 3):
                    issues.append(f"Invalid image shape at {idx}: {img.shape}")
                    continue
                
                if np.isnan(img).any() or np.isinf(img).any():
                    issues.append(f"NaN/Inf values in image at {idx}")
                    continue
                
                if img.min() < 0 or img.max() > 255:
                    issues.append(f"Invalid pixel values at {idx}: [{img.min()}, {img.max()}]")
                    continue
                
                # Check label
                label = self.y_data[idx, 0, 0, 0]
                if label not in [0, 1]:
                    issues.append(f"Invalid label at {idx}: {label}")
            
            except Exception as e:
                issues.append(f"Error checking sample {idx}: {str(e)}")
        
        if issues:
            logger.error(f"Dataset validation found {len(issues)} issues:")
            for issue in issues[:10]:  # Log first 10 issues
                logger.error(f"  - {issue}")
            raise ValueError(f"Dataset validation failed with {len(issues)} issues")
        
        logger.info(f"Dataset validation passed ({n_samples_to_check} samples checked)")
    
    def _compute_class_distribution(self) -> Dict[int, int]:
        """Compute class counts for monitoring and loss weighting."""
        labels = self.y_data[:, 0, 0, 0]
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique.astype(int), counts.astype(int)))
        
        # Ensure both classes are present
        if 0 not in distribution:
            distribution[0] = 0
        if 1 not in distribution:
            distribution[1] = 0
        
        return distribution
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for weighted loss.
        
        Uses inverse frequency: weight = N / (n_classes * count)
        For PCam ~60/40 split, this gives reasonable weights.
        """
        n_classes = 2
        weights = np.zeros(n_classes)
        
        for cls, count in self.class_counts.items():
            weights[cls] = self.n_samples / (n_classes * count)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_sample_weights(self) -> np.ndarray:
        """
        Get per-sample weights for weighted sampling.
        
        Useful for oversampling minority class during training.
        """
        class_weights = self.get_class_weights().numpy()
        
        # Assign weight to each sample based on its class
        sample_weights = np.zeros(self.n_samples)
        for idx in range(self.n_samples):
            label = int(self.y_data[idx, 0, 0, 0])
            sample_weights[idx] = class_weights[label]
        
        return sample_weights
    
    def get_statistics(self) -> Dict[str, float]:
        """Compute dataset statistics for monitoring drift."""
        # Sample subset for efficiency
        n_samples = min(10000, self.n_samples)
        indices = np.random.choice(self.n_samples, n_samples, replace=False)
        
        images = self.x_data[indices]
        
        stats = {
            'mean': float(images.mean()),
            'std': float(images.std()),
            'min': float(images.min()),
            'max': float(images.max()),
            'median': float(np.median(images)),
            'q25': float(np.percentile(images, 25)),
            'q75': float(np.percentile(images, 75)),
        }
        
        return stats
    
    def __del__(self):
        """
        Close H5 files on deletion.
        
        CRITICAL: Ensures proper resource cleanup to prevent file handle leaks
        during long-running training or batch processing.
        """
        try:
            if hasattr(self, 'x_file') and self.x_file is not None:
                self.x_file.close()
                logger.debug(f"Closed H5 file: {self.x_path.name}")
        except Exception as e:
            logger.warning(f"Error closing x_file: {e}")
        
        try:
            if hasattr(self, 'y_file') and self.y_file is not None:
                self.y_file.close()
                logger.debug(f"Closed H5 file: {self.y_path.name}")
        except Exception as e:
            logger.warning(f"Error closing y_file: {e}")
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  n_samples={self.n_samples},\n"
            f"  positive_ratio={self.positive_ratio:.2%},\n"
            f"  stain_normalized={self.stain_normalizer is not None},\n"
            f"  augmentation={self.transform is not None}\n"
            f")"
        )


class PCamTestTimeAugmentationDataset(Dataset):
    """
    Dataset wrapper for Test-Time Augmentation (TTA).
    
    Applies multiple augmentations to each test sample and returns all variants.
    Predictions are aggregated (mean/voting) for improved robustness.
    
    Medical Rationale:
    - Tissue has no canonical orientation
    - TTA reduces prediction variance
    - Particularly effective for borderline cases
    """
    
    def __init__(
        self,
        base_dataset: PCamDataset,
        tta_transforms: List[Callable],
    ):
        self.base_dataset = base_dataset
        self.tta_transforms = tta_transforms
        self.n_augmentations = len(tta_transforms)
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            images: Tensor of shape (n_augmentations, 3, 96, 96)
            label: Tensor of shape (1,)
        """
        # Get original image and label
        base_image, label = self.base_dataset[idx]
        
        # Apply all TTA transforms
        augmented_images = []
        for transform in self.tta_transforms:
            if isinstance(base_image, torch.Tensor):
                img = base_image.detach().cpu()
                if img.dim() == 3:
                    img = img.permute(1, 2, 0)
                img = (img * 255.0).clamp(0, 255).numpy().astype(np.uint8)
                base_pil = Image.fromarray(img)
            else:
                base_pil = base_image
            aug_image = transform(base_pil)
            augmented_images.append(aug_image)
        
        # Stack all augmentations
        images = torch.stack(augmented_images)
        
        return images, label


def create_dataloaders(
    config: Dict,
    train_dataset: PCamDataset,
    val_dataset: PCamDataset,
    test_dataset: Optional[PCamDataset] = None,
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Factory function to create train/val/test dataloaders.
    
    Implements medical-grade data loading:
    - Stratified sampling to maintain class balance
    - Efficient prefetching for GPU utilization
    - Deterministic shuffling for reproducibility
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler
    
    # Training dataloader with optional weighted sampling
    if config.get('use_weighted_sampling', False):
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        shuffle = True
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.get('num_workers', 8),
        pin_memory=config.get('pin_memory', True),
        drop_last=True,
        prefetch_factor=config.get('prefetch_factor', 2),
        persistent_workers=config.get('persistent_workers', True),
    )
    
    # Validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['val_batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 8),
        pin_memory=config.get('pin_memory', True),
        drop_last=False,
    )
    
    # Test dataloader
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['test_batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 8),
            pin_memory=config.get('pin_memory', True),
            drop_last=False,
        )
        return train_loader, val_loader, test_loader
    
    return train_loader, val_loader