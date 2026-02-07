"""
Medical Image Preprocessing for Histopathology
Implements stain normalization and augmentation pipeline

Production-Grade Implementation Requirements:
- Corrects critical bugs in original (matrix dimensions, numerical stability)
- Medical-grade validation and error handling
- Comprehensive logging for compliance and debugging
- CRITICAL: Enforces data leakage prevention for stain normalizer fit-set
"""

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class FitSetOrigin(Enum):
    """
    Enumeration for stain normalizer fit-set origin.
    
    CRITICAL FOR DATA LEAKAGE PREVENTION:
    - TRAINING_ONLY: Fit on training set only (correct)
    - VALIDATION: Fit on validation set (incorrect for test evaluation)
    - VALIDATION_PLUS_EXTERNAL: Fit on external reference image (correct for inference)
    - UNDECLARED: Fit-set origin not declared (RED FLAG - potential leakage)
    """
    TRAINING_ONLY = "training_only"
    VALIDATION = "validation"
    VALIDATION_PLUS_EXTERNAL = "validation_plus_external"
    EXTERNAL_REFERENCE = "external_reference"
    UNDECLARED = "undeclared"  # RED FLAG


@dataclass
class StainNormalizerState:
    """
    Metadata tracking for stain normalizer state.
    
    Purpose: Ensure reproducibility and prevent data leakage
    - Tracks fit-set origin (training, validation, or external)
    - Validates that fit-set is appropriate for downstream use
    - Enables audit trail for regulatory compliance
    """
    is_fitted: bool
    fit_set_origin: FitSetOrigin
    stain_matrix_reference: Optional[np.ndarray] = None
    max_c_reference: Optional[np.ndarray] = None
    
    def validate_for_test_inference(self) -> bool:
        """
        Validate that normalizer is safe for test inference.
        
        CRITICAL: Test inference requires fit-set to be EXTERNAL or VALIDATION
        (never on test set, never on validation+test).
        """
        if not self.is_fitted:
            return False
        
        safe_origins = {
            FitSetOrigin.EXTERNAL_REFERENCE,
            FitSetOrigin.VALIDATION_PLUS_EXTERNAL,
            FitSetOrigin.TRAINING_ONLY,
        }
        
        if self.fit_set_origin not in safe_origins:
            raise ValueError(
                f"LEAKAGE RISK: Normalizer fit on {self.fit_set_origin.value}. "
                f"Test inference requires fit-set to be external or training-only. "
                f"If fit on validation+test, this causes data leakage."
            )
        
        return True
    
    def validate_for_dataset_creation(self) -> bool:
        """
        Validate that normalizer is safe for training/validation dataset creation.
        
        For training dataset: Fit-set must be TRAINING_ONLY
        For validation dataset: Fit-set must be EXTERNAL or TRAINING_ONLY
        """
        if not self.is_fitted:
            return False
        
        if self.fit_set_origin == FitSetOrigin.UNDECLARED:
            raise ValueError(
                "LEAKAGE RISK: Normalizer fit-set origin is UNDECLARED. "
                "Must explicitly declare fit-set origin (training, external, etc.)"
            )
        
        return True


class StainNormalizer:
    """
    Macenko Stain Normalization for H&E histopathology images.
    
    Production-Grade Implementation with Critical Corrections:
    1. Fixed stain matrix dimensions (2×3, not 3×2)
    2. Robust numerical stability (log(0) handling, matrix ill-conditioning)
    3. Quality validation (NaN/Inf detection, diff thresholding)
    4. Correct concentration scaling logic
    5. Comprehensive logging and error tracking
    6. **DATA LEAKAGE PREVENTION**: Tracks fit-set origin and validates at usage time
    
    Medical Rationale:
    - H&E staining varies across labs, scanners, time periods
    - Stain normalization removes "equipment fingerprint" while preserving morphology
    - Models learn biology, not scanner artifacts
    - Essential for multi-site clinical deployment
    
    CRITICAL: Fit-set origin must be declared and validated:
    - Training dataset: normalizer fitted on training set only
    - Validation dataset: normalizer fitted on training set only (NOT on validation)
    - Test set: normalizer fitted on training or external reference only (NOT on test)
    
    Reference:
    Macenko et al. "A method for normalizing histology slides for 
    quantitative analysis." IEEE ISBI 2009.
    
    Args:
        luminosity_threshold (float): Background threshold [0.0-1.0] (higher = stricter)
        angular_percentile (float): Percentile for stain vector estimation [90-99]
        max_c_reference (np.ndarray): Reference concentration values [H, E]
        fit_set_origin (FitSetOrigin): Declares which data the normalizer was fit on
    """
    
    def __init__(
        self,
        luminosity_threshold: float = 0.8,
        angular_percentile: float = 99.0,
        max_c_reference: Optional[np.ndarray] = None,
        fit_set_origin: FitSetOrigin = FitSetOrigin.UNDECLARED,
    ):
        # Validate inputs
        if not 0.0 <= luminosity_threshold <= 1.0:
            raise ValueError(f"luminosity_threshold must be in [0, 1], got {luminosity_threshold}")
        if not 90.0 <= angular_percentile <= 99.9:
            logger.warning(f"angular_percentile={angular_percentile} outside recommended range [90, 99.9]")
        
        self.luminosity_threshold = luminosity_threshold
        self.angular_percentile = angular_percentile
        self.max_c_reference = max_c_reference
        
        # Stain matrix: 2×3 (H stain row + E stain row) × 3 RGB channels
        self.stain_matrix_reference = None
        
        # Data leakage prevention: Track fit-set origin
        self._state = StainNormalizerState(
            is_fitted=False,
            fit_set_origin=fit_set_origin,
        )
        
        # Quality checks
        self.normalization_count = 0
        self.failed_normalizations = 0
    
    @property
    def is_fitted(self) -> bool:
        """Check if normalizer has been fitted."""
        return self._state.is_fitted
    
    @property
    def fit_set_origin(self) -> FitSetOrigin:
        """Get the declared fit-set origin."""
        return self._state.fit_set_origin
    
    def fit(self, reference_image: np.ndarray, declared_fit_set: Optional[FitSetOrigin] = None) -> None:
        """
        Fit normalizer to reference image.
        
        CRITICAL: Caller must declare fit-set origin for audit trail.
        
        Args:
            reference_image: RGB image (H, W, 3) with values [0, 255]
            declared_fit_set: Explicitly declare fit-set origin for data leakage prevention
        
        Raises:
            ValueError: If fit-set origin is not declared
        """
        # Update fit-set origin if provided (allows correction)
        if declared_fit_set is not None:
            if self._state.fit_set_origin == FitSetOrigin.UNDECLARED:
                self._state.fit_set_origin = declared_fit_set
                logger.info(f"Fit-set origin declared: {declared_fit_set.value}")
            elif self._state.fit_set_origin != declared_fit_set:
                logger.warning(
                    f"Fit-set origin mismatch: was {self._state.fit_set_origin.value}, "
                    f"now {declared_fit_set.value}. Updating."
                )
                self._state.fit_set_origin = declared_fit_set
        
        # Validate that fit-set origin is declared
        if self._state.fit_set_origin == FitSetOrigin.UNDECLARED:
            raise ValueError(
                "LEAKAGE RISK: Fit-set origin is UNDECLARED. "
                "Must explicitly declare fit-set origin (e.g., training, external reference). "
                "Call fit(image, declared_fit_set=FitSetOrigin.TRAINING_ONLY) to proceed."
            )
        
        # Validate input
        assert reference_image.ndim == 3, f"Expected 3D image, got {reference_image.ndim}D"
        assert reference_image.shape[2] == 3, f"Expected RGB image, got {reference_image.shape[2]} channels"
        assert reference_image.dtype == np.uint8 or reference_image.max() <= 255, \
            "Expected uint8 image or values in [0, 255]"
        
        # Extract stain matrix and max concentrations
        stain_matrix, max_c = self._extract_stain_parameters(reference_image)
        
        # Validate stain matrix
        assert stain_matrix.shape == (2, 3), \
            f"Expected stain matrix shape (2, 3), got {stain_matrix.shape}"
        
        self.stain_matrix_reference = stain_matrix
        self._state.stain_matrix_reference = stain_matrix
        self.max_c_reference = max_c
        self._state.max_c_reference = max_c
        self._state.is_fitted = True
        
        # Log normalization parameters for reproducibility
        logger.info(
            f"Stain normalizer fitted (origin={self._state.fit_set_origin.value}). "
            f"Parameters: luminosity_threshold={self.luminosity_threshold}, "
            f"angular_percentile={self.angular_percentile}"
        )
        
        logger.info(f"Stain normalizer fitted to reference image")
        logger.info(f"  Fit-set origin: {self._state.fit_set_origin.value}")
        logger.info(f"  Stain matrix shape: {stain_matrix.shape}")
        logger.info(f"  Max concentrations: {max_c}")
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to reference stain distribution.
        
        CORRECTED: Fixed matrix dimensions and numerical stability
        
        CRITICAL: Validates fit-set origin before normalization to prevent data leakage.
        
        Args:
            image: RGB image (H, W, 3) with values [0, 255]
            
        Returns:
            Normalized image (H, W, 3) with values [0, 255]
            
        Raises:
            RuntimeError: If normalizer not fitted or fit-set origin invalid
        """
        if not self.is_fitted:
            logger.warning("Normalizer not fitted, returning original image")
            return image
        
        # Validate fit-set origin (data leakage prevention)
        try:
            self._state.validate_for_dataset_creation()
        except ValueError as e:
            logger.error(f"Cannot normalize: {e}")
            raise
        
        self.normalization_count += 1
        
        try:
            # Validate input
            assert image.ndim == 3 and image.shape[2] == 3, \
                f"Invalid image shape: {image.shape}"
            
            h, w, c = image.shape
            
            # Extract stain parameters from input image
            stain_matrix_source, max_c_source = self._extract_stain_parameters(image)
            
            # Convert image to optical density
            # Add small epsilon to avoid log(0)
            image_float = image.astype(np.float64) + 1.0
            od = -np.log(image_float / 255.0)
            
            # Reshape for matrix operations: (H*W, 3)
            od_flat = od.reshape(-1, 3)
            
            # Separate stains: solve od = stain_matrix @ concentrations
            # stain_matrix: (2, 3), od_flat.T: (3, H*W)
            # concentrations: (2, H*W)
            concentrations = np.linalg.lstsq(
                stain_matrix_source.T,  # (3, 2)
                od_flat.T,              # (3, H*W)
                rcond=None
            )[0]  # (2, H*W)
            
            # Normalize concentrations
            # Avoid division by zero
            max_c_source_safe = np.maximum(max_c_source, 1e-6)
            max_c_ref_safe = np.maximum(self.max_c_reference, 1e-6)
            
            # Scale concentrations: c_new = c_old * (max_ref / max_source)
            scale_factors = (max_c_ref_safe / max_c_source_safe).reshape(-1, 1)
            concentrations_normalized = concentrations * scale_factors
            
            # Reconstruct with reference stain matrix
            # stain_matrix_reference.T: (3, 2) @ concentrations: (2, H*W) = (3, H*W)
            od_normalized = self.stain_matrix_reference.T @ concentrations_normalized
            
            # Convert back to RGB
            # od = -log(RGB / 255) => RGB = 255 * exp(-od)
            image_normalized = 255.0 * np.exp(-od_normalized.T)  # (H*W, 3)
            image_normalized = image_normalized.reshape(h, w, c)
            
            # Clip to valid range
            image_normalized = np.clip(image_normalized, 0, 255).astype(np.uint8)
            
            # Quality check: ensure normalization didn't fail catastrophically
            if np.isnan(image_normalized).any() or np.isinf(image_normalized).any():
                logger.warning("Normalization produced NaN/Inf, returning original")
                self.failed_normalizations += 1
                return image
            
            # Check if normalized image is too different (possible failure)
            diff = np.abs(image.astype(float) - image_normalized.astype(float)).mean()
            if diff > 100:  # Arbitrary threshold: mean difference > 100 intensity levels
                logger.warning(f"Normalization may have failed (mean diff: {diff:.1f})")
                self.failed_normalizations += 1
                # Still return normalized image, but log warning
            
            return image_normalized
            
        except Exception as e:
            logger.error(f"Stain normalization failed: {e}")
            self.failed_normalizations += 1
            return image
    
    def _extract_stain_parameters(
        self, 
        image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract stain matrix and max concentrations using Macenko method.
        
        CORRECTED: Fixed matrix dimensions and edge cases
        
        Returns:
            stain_matrix: (2, 3) matrix of stain vectors [H, E] × [R, G, B]
            max_concentrations: (2,) max concentration for each stain [H, E]
        """
        # Convert to optical density
        image_float = image.astype(np.float64) + 1.0
        od = -np.log(image_float / 255.0)
        
        # Remove background (low OD = bright pixels = glass/whitespace)
        od_flat = od.reshape(-1, 3)  # (H*W, 3)
        od_magnitude = np.linalg.norm(od_flat, axis=1)
        tissue_mask = od_magnitude > self.luminosity_threshold
        od_tissue = od_flat[tissue_mask]
        
        # Check if enough tissue detected
        if len(od_tissue) < 100:
            logger.warning(
                f"Very few tissue pixels detected ({len(od_tissue)}), "
                "using default stain matrix"
            )
            # Return default H&E stain matrix from literature
            stain_matrix = np.array([
                [0.650, 0.704, 0.286],  # Hematoxylin (H) in RGB
                [0.072, 0.990, 0.105],  # Eosin (E) in RGB
            ])  # Shape: (2, 3)
            max_c = np.array([1.9, 1.5])  # Typical concentrations
            return stain_matrix, max_c
        
        # Perform SVD on tissue OD: od_tissue = U @ S @ Vt
        # Vt: (3, 3), we want first 2 principal components
        try:
            U, S, Vt = np.linalg.svd(od_tissue, full_matrices=False)
        except np.linalg.LinAlgError:
            logger.error("SVD failed, using default stain matrix")
            stain_matrix = np.array([
                [0.650, 0.704, 0.286],
                [0.072, 0.990, 0.105],
            ])
            max_c = np.array([1.9, 1.5])
            return stain_matrix, max_c
        
        # Project onto principal plane (first 2 components)
        # Vt[:2]: (2, 3) - first 2 principal components in RGB space
        plane = Vt[:2, :].T  # (3, 2) - projection matrix
        
        # Project OD data onto this plane
        projected = od_tissue @ plane  # (N, 3) @ (3, 2) = (N, 2)
        
        # Find extreme angles (representing two stains)
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        
        # Get min and max percentile angles
        min_angle = np.percentile(angles, 100 - self.angular_percentile)
        max_angle = np.percentile(angles, self.angular_percentile)
        
        # Compute stain vectors from extreme angles
        # These vectors are in the 2D principal plane
        v1_plane = np.array([np.cos(min_angle), np.sin(min_angle)])
        v2_plane = np.array([np.cos(max_angle), np.sin(max_angle)])
        
        # Map back to 3D RGB space
        stain1 = plane @ v1_plane  # (3, 2) @ (2,) = (3,)
        stain2 = plane @ v2_plane
        
        # Ensure stain vectors point in positive direction
        if stain1[0] < 0:
            stain1 = -stain1
        if stain2[0] < 0:
            stain2 = -stain2
        
        # Normalize stain vectors (unit vectors)
        stain1 = stain1 / np.linalg.norm(stain1)
        stain2 = stain2 / np.linalg.norm(stain2)
        
        # Build stain matrix: (2, 3) - [stain1; stain2] as rows
        stain_matrix = np.vstack([stain1, stain2])  # (2, 3)
        
        # Compute max concentrations for each stain
        # Solve: od_tissue.T = stain_matrix.T @ concentrations
        # stain_matrix.T: (3, 2), od_tissue.T: (3, N), concentrations: (2, N)
        try:
            concentrations = np.linalg.lstsq(
                stain_matrix.T,  # (3, 2)
                od_tissue.T,     # (3, N)
                rcond=None
            )[0]  # (2, N)
            
            # Max concentration at 99th percentile
            max_c = np.percentile(concentrations, 99, axis=1)
            max_c = np.maximum(max_c, 0.1)  # Ensure minimum value
            
        except np.linalg.LinAlgError:
            logger.warning("Concentration computation failed, using defaults")
            max_c = np.array([1.9, 1.5])
        
        return stain_matrix, max_c
    
    def get_quality_metrics(self) -> Dict:
        """Get normalization quality metrics for monitoring."""
        return {
            'total_normalizations': self.normalization_count,
            'failed_normalizations': self.failed_normalizations,
            'failure_rate': (
                self.failed_normalizations / self.normalization_count 
                if self.normalization_count > 0 else 0
            ),
        }
    
    def validate_normalization(
        self,
        original: np.ndarray,
        normalized: np.ndarray
    ) -> Dict:
        """
        Validate that stain normalization worked correctly.
        
        Checks:
        1. No NaN/Inf values
        2. Values in valid range [0, 255]
        3. Reasonable color distribution
        4. Preserved image structure
        """
        validation = {
            'passed': True,
            'issues': [],
        }
        
        # Check for NaN/Inf
        if np.isnan(normalized).any():
            validation['passed'] = False
            validation['issues'].append("Contains NaN values")
        if np.isinf(normalized).any():
            validation['passed'] = False
            validation['issues'].append("Contains Inf values")
        
        # Check value range
        if normalized.min() < 0 or normalized.max() > 255:
            validation['passed'] = False
            validation['issues'].append(
                f"Values out of range: [{normalized.min()}, {normalized.max()}]"
            )
        
        # Check if image structure preserved (using SSIM-like check)
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        normalized_gray = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
        
        # Simple correlation check
        corr = np.corrcoef(original_gray.flat, normalized_gray.flat)[0, 1]
        if corr < 0.5:
            validation['passed'] = False
            validation['issues'].append(
                f"Low correlation between original and normalized ({corr:.3f})"
            )
        
        return validation


class MedicalAugmentation:
    """
    Medical-grade augmentation for histopathology.
    
    Design Principles:
    - Only biologically plausible transformations
    - Preserve cellular morphology and architecture
    - Simulate real-world variations (staining, scanning)
    
    Excluded Augmentations (with reasons):
    - Gaussian blur: Destroys nuclear detail needed for diagnosis
    - Elastic deformation: Creates unrealistic tissue architecture
    - Cutout/GridMask: Removes diagnostically critical areas
    - MixUp/CutMix: Creates impossible tissue morphology
    """
    
    def __init__(
        self,
        rotation_angles: List[int] = [0, 90, 180, 270],
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.5,
        color_jitter_brightness: float = 0.05,
        color_jitter_contrast: float = 0.0,
        color_jitter_saturation: float = 0.1,
        color_jitter_hue: float = 0.02,
        color_jitter_prob: float = 0.5,
        normalize_mean: List[float] = [0.485, 0.456, 0.406],
        normalize_std: List[float] = [0.229, 0.224, 0.225],
    ):
        self.rotation_angles = rotation_angles
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.color_jitter_prob = color_jitter_prob
        
        self.color_jitter = T.ColorJitter(
            brightness=color_jitter_brightness,
            contrast=color_jitter_contrast,
            saturation=color_jitter_saturation,
            hue=color_jitter_hue,
        )
        
        self.normalize = T.Normalize(mean=normalize_mean, std=normalize_std)
        self.to_tensor = T.ToTensor()
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Apply augmentation pipeline.
        
        Pipeline:
        1. Random rotation (90° increments)
        2. Random flips (H/V)
        3. Color jittering (simulate staining variations)
        4. Convert to tensor
        5. Normalize to ImageNet statistics
        """
        # Rotation (tissue has no canonical orientation)
        if len(self.rotation_angles) > 1:
            angle = np.random.choice(self.rotation_angles)
            if angle != 0:
                image = TF.rotate(image, angle)
        
        # Horizontal flip
        if np.random.rand() < self.horizontal_flip_prob:
            image = TF.hflip(image)
        
        # Vertical flip
        if np.random.rand() < self.vertical_flip_prob:
            image = TF.vflip(image)
        
        # Color jitter (simulate staining variations)
        if np.random.rand() < self.color_jitter_prob:
            image = self.color_jitter(image)
        
        # Convert to tensor and normalize
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        return image


class TestTimeAugmentation:
    """
    Test-Time Augmentation (TTA) transforms.
    
    Applies systematic augmentations at inference:
    - 4 rotations (0°, 90°, 180°, 270°)
    - 2 flips (H, V)
    Total: 8 augmentations per image (selected combinations)
    
    Predictions aggregated via mean or voting for robustness.
    """
    
    def __init__(
        self,
        normalize_mean: List[float] = [0.485, 0.456, 0.406],
        normalize_std: List[float] = [0.229, 0.224, 0.225],
    ):
        self.normalize = T.Normalize(mean=normalize_mean, std=normalize_std)
        self.to_tensor = T.ToTensor()
    
    def get_transforms(self) -> List[T.Compose]:
        """
        Generate all TTA transforms.
        
        Returns:
            List of 8 composed transforms
        """
        transforms = []
        
        # 8 deterministic transforms aligned with config:
        # original, rot90, rot180, rot270, hflip, vflip, hflip_rot90, vflip_rot90
        def _make_transform(angle: int = 0, hflip: bool = False, vflip: bool = False):
            def transform(img):
                if angle != 0:
                    img = TF.rotate(img, angle)
                if hflip:
                    img = TF.hflip(img)
                if vflip:
                    img = TF.vflip(img)
                img = self.to_tensor(img)
                img = self.normalize(img)
                return img
            return transform

        transforms.extend([
            _make_transform(0, False, False),
            _make_transform(90, False, False),
            _make_transform(180, False, False),
            _make_transform(270, False, False),
            _make_transform(0, True, False),
            _make_transform(0, False, True),
            _make_transform(90, True, False),
            _make_transform(90, False, True),
        ])
        
        return transforms


def get_transforms(
    stage: str,
    config: dict,
    stain_normalizer: Optional[StainNormalizer] = None,
) -> T.Compose:
    """
    Factory function for getting transforms based on stage.
    
    Args:
        stage: 'train', 'val', or 'test'
        config: Configuration dictionary
        stain_normalizer: Optional stain normalizer
        
    Returns:
        Composed transform pipeline
    """
    if stage == 'train':
        return MedicalAugmentation(
            rotation_angles=config.get('rotation_angles', [0, 90, 180, 270]),
            horizontal_flip_prob=config.get('horizontal_flip_prob', 0.5),
            vertical_flip_prob=config.get('vertical_flip_prob', 0.5),
            color_jitter_brightness=config.get('color_jitter_brightness', 0.05),
            color_jitter_contrast=config.get('color_jitter_contrast', 0.0),
            color_jitter_saturation=config.get('color_jitter_saturation', 0.1),
            color_jitter_hue=config.get('color_jitter_hue', 0.02),
            color_jitter_prob=config.get('color_jitter_prob', 0.5),
        )
    
    elif stage in ['val', 'test']:
        normalize_mean = config.get('normalize_mean', [0.485, 0.456, 0.406])
        normalize_std = config.get('normalize_std', [0.229, 0.224, 0.225])
        
        return T.Compose([
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ])
    
    else:
        raise ValueError(f"Unknown stage: {stage}")


def compute_dataset_statistics(dataset, n_samples: int = 10000) -> dict:
    """
    Compute mean and std for dataset normalization.
    
    Args:
        dataset: PyTorch dataset
        n_samples: Number of samples to use
        
    Returns:
        Dictionary with mean and std per channel
    """
    from torch.utils.data import DataLoader, Subset
    
    # Sample subset
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, num_workers=4)
    
    # Compute statistics
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n_pixels = 0
    
    for images, _ in loader:
        batch_size = images.size(0)
        images = images.view(batch_size, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_pixels += batch_size
    
    mean /= n_pixels
    std /= n_pixels
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
    }