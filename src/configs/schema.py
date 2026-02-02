"""
Unified Configuration Schema for PCam Classification

CRITICAL: All parameters must be defined in config, not hardcoded in code.
This ensures reproducibility, auditability, and regulatory compliance.

Configuration Hierarchy:
- Base: Pydantic models with strict validation
- YAML files: Populated from Pydantic models
- Runtime: Config validated at startup, immutable after

Validation:
- Type checking (int, float, str, Enum)
- Range validation (e.g., learning_rate > 0)
- Consistency checks (e.g., val_split < 1.0)
- Required field enforcement

FDA Compliance:
- All parameters logged for audit trail
- Version tracking for configuration changes
- Deterministic behavior from config
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, List, Optional, Literal, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OptimiserType(str, Enum):
    """Supported optimizers."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    LAMB = "lamb"


class SchedulerType(str, Enum):
    """Supported learning rate schedulers."""
    STEPLR = "steplr"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    PLATEAU = "plateau"


class LossType(str, Enum):
    """Supported loss functions."""
    BCE = "bce"
    FOCAL = "focal"
    WEIGHTED_BCE = "weighted_bce"
    ASYMMETRIC = "asymmetric"
    CLINICAL = "clinical"


class AugmentationType(str, Enum):
    """Augmentation strategies."""
    MEDICAL_GRADE = "medical_grade"  # Rotation, flip, color jitter
    AGGRESSIVE = "aggressive"         # Includes elastic deformation
    MINIMAL = "minimal"               # Only horizontal/vertical flip


class OptimizerConfig(BaseModel):
    """Optimizer configuration."""
    type: OptimiserType = OptimiserType.ADAMW
    learning_rate: float = Field(1e-4, gt=0, le=1.0)
    weight_decay: float = Field(1e-5, ge=0, le=1.0)
    betas: tuple = Field((0.9, 0.999), description="Beta parameters for Adam-like optimizers")
    eps: float = Field(1e-8, gt=0)

    class Config:
        use_enum_values = False


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""
    type: SchedulerType = SchedulerType.COSINE
    warmup_epochs: int = Field(0, ge=0)
    step_size: Optional[int] = Field(10, gt=0)
    gamma: float = Field(0.1, gt=0, le=1.0)
    t_max: Optional[int] = None
    patience: int = Field(5, gt=0)

    class Config:
        use_enum_values = False


class LossConfig(BaseModel):
    """Loss function configuration."""
    type: LossType = LossType.FOCAL
    focal_alpha: float = Field(0.25, ge=0, le=1.0)
    focal_gamma: float = Field(2.0, gt=0)
    pos_weight: float = Field(1.5, gt=0)
    fn_weight: float = Field(3.0, gt=0, description="Weight for false negatives in clinical loss")

    class Config:
        use_enum_values = False


class TrainingConfig(BaseModel):
    """Complete training configuration."""
    batch_size: int = Field(32, gt=0)
    val_batch_size: int = Field(64, gt=0)
    epochs: int = Field(50, gt=0)
    seed: int = Field(42, ge=0)
    
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    
    mixed_precision: bool = True
    gradient_accumulation_steps: int = Field(1, gt=0)
    
    early_stopping_patience: int = Field(15, gt=0)
    early_stopping_min_delta: float = Field(0.0001, ge=0)
    
    num_workers: int = Field(8, ge=0)
    pin_memory: bool = True
    persistent_workers: bool = True
    
    checkpoint_dir: str = Field("experiments/checkpoints")
    save_best_only: bool = True
    
    # Validation
    use_weighted_sampling: bool = False
    
    @validator('batch_size', 'val_batch_size')
    def batch_size_must_fit_gpu(cls, v):
        """Validate batch size is reasonable for typical GPUs."""
        if v > 512:
            logger.warning(f"Large batch size {v}. May cause OOM on typical GPUs.")
        return v
    
    @root_validator
    def validate_training(cls, values):
        """Cross-field validation."""
        if values.get('batch_size', 32) > values.get('val_batch_size', 64):
            logger.info("Training batch size > validation batch size (expected)")
        return values

    class Config:
        use_enum_values = False


class AugmentationConfig(BaseModel):
    """Data augmentation configuration."""
    strategy: AugmentationType = AugmentationType.MEDICAL_GRADE
    
    rotation_angles: List[int] = Field([0, 90, 180, 270])
    horizontal_flip: bool = True
    vertical_flip: bool = True
    
    color_jitter: bool = True
    brightness: float = Field(0.2, ge=0)
    contrast: float = Field(0.2, ge=0)
    saturation: float = Field(0.2, ge=0)
    hue: float = Field(0.1, ge=0)
    
    # Medical-specific
    elastic_deformation: bool = False
    elastic_alpha: float = Field(30.0, gt=0)
    elastic_sigma: float = Field(3.0, gt=0)
    
    # Mixup/Cutmix
    mixup_alpha: float = Field(0.0, ge=0, description="0 = disabled")
    cutmix_alpha: float = Field(0.0, ge=0, description="0 = disabled")

    class Config:
        use_enum_values = False


class StainNormalizationConfig(BaseModel):
    """Stain normalization configuration."""
    enabled: bool = True
    method: Literal["macenko", "reinhard"] = "macenko"
    
    # Macenko parameters
    luminosity_threshold: float = Field(0.8, ge=0.0, le=1.0)
    angular_percentile: float = Field(99.0, ge=90.0, le=99.9)
    
    # Reference image
    reference_image_path: Optional[str] = Field(None, description="Path to reference H&E image")
    
    # Numerical stability
    epsilon: float = Field(1e-6, gt=0)

    @validator('luminosity_threshold')
    def validate_luminosity(cls, v):
        """Validate luminosity threshold."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"luminosity_threshold must be in [0, 1], got {v}")
        return v

    class Config:
        use_enum_values = False


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration."""
    data_dir: str = Field("data")
    
    # H5 file paths
    train_x: str = "train_images.h5"
    train_y: str = "train_labels.h5"
    val_x: str = "valid_images.h5"
    val_y: str = "valid_labels.h5"
    test_x: Optional[str] = Field(None, description="Test set (optional, held-out)")
    test_y: Optional[str] = None
    
    # Splits
    train_split: float = Field(0.7, gt=0, lt=1.0, description="Fraction for training")
    val_split: float = Field(0.15, gt=0, lt=1.0, description="Fraction for validation")
    
    # Preprocessing
    stain_normalization: StainNormalizationConfig = Field(default_factory=StainNormalizationConfig)
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    
    # Image statistics (computed from training set)
    normalization_mean: List[float] = Field([0.485, 0.456, 0.406])
    normalization_std: List[float] = Field([0.229, 0.224, 0.225])
    
    # Cache
    cache_normalized: bool = False
    cache_dir: Optional[str] = Field(None, description="Cache directory for normalized images")
    
    # Validation
    validate_data: bool = True
    check_duplicates: bool = True
    max_duplicate_check_samples: int = Field(2000, gt=0, description="Sample size for duplicate detection")
    
    seed: int = Field(42, ge=0)
    
    @root_validator
    def validate_splits(cls, values):
        """Validate train/val splits sum to <= 1.0."""
        train = values.get('train_split', 0.7)
        val = values.get('val_split', 0.15)
        if train + val > 1.0:
            raise ValueError(f"train_split + val_split must be <= 1.0, got {train + val}")
        return values

    class Config:
        use_enum_values = False


class ModelArchitecture(str, Enum):
    """Supported model architectures."""
    RESNET50 = "resnet50"
    RESNET50_CBAM = "resnet50_cbam"
    EFFICIENTNET_B3 = "efficientnet_b3"
    VIT_B16 = "vit_b16"
    DEIT_SMALL = "deit_small"


class ModelConfig(BaseModel):
    """Model architecture configuration."""
    architecture: ModelArchitecture = ModelArchitecture.RESNET50
    
    pretrained: bool = True
    pretrained_weights: Literal["imagenet", "imagenet21k"] = "imagenet"
    
    # Freezing
    freeze_backbone: bool = True
    freeze_until_layer: Optional[int] = Field(None, description="Freeze until layer N (0-indexed)")
    
    # Architecture-specific
    dropout: float = Field(0.3, ge=0, le=1.0)
    hidden_dims: List[int] = Field([512, 256], description="Hidden dimensions in classifier")
    num_classes: int = Field(1, ge=1, description="1 for binary sigmoid, 2 for softmax")
    
    # Attention parameters
    se_reduction: int = Field(16, gt=0, description="Squeeze-and-excitation reduction factor")
    use_spatial_attention: bool = True
    use_dual_pooling: bool = True
    
    # Output parameters
    use_logits: bool = True

    class Config:
        use_enum_values = False


class EnsembleConfig(BaseModel):
    """Ensemble configuration."""
    enabled: bool = False
    
    models: List[ModelConfig] = Field(default_factory=list)
    weights: Optional[List[float]] = Field(None, description="Ensemble weights (auto-normalized)")
    
    aggregation: Literal["soft_voting", "stacking"] = "soft_voting"
    
    @validator('weights')
    def validate_weights(cls, v, values):
        """Validate ensemble weights."""
        if v is None:
            return None
        if len(v) != len(values.get('models', [])):
            raise ValueError(f"weights length must match models length")
        if any(w < 0 for w in v):
            raise ValueError("Weights must be non-negative")
        return v

    class Config:
        use_enum_values = False


class InferenceConfig(BaseModel):
    """Inference configuration."""
    batch_size: int = Field(16, gt=0)
    
    # Test-time augmentation
    use_tta: bool = False
    tta_transforms: List[str] = Field(["identity"], description="TTA augmentation strategies")
    
    # Threshold
    threshold: float = Field(0.5, ge=0, le=1.0)
    
    # Calibration
    calibration_enabled: bool = True
    calibration_path: Optional[str] = Field(None, description="Path to calibration file")
    
    # Export
    export_formats: List[str] = Field(["pytorch"], description="Formats: pytorch, onnx, torchscript, quantized")
    quantization_type: Literal["none", "int8", "fp16"] = "none"

    class Config:
        use_enum_values = False


class APIConfig(BaseModel):
    """API service configuration."""
    host: str = "0.0.0.0"
    port: int = Field(8000, gt=0, lt=65536)
    
    # Security
    api_key_enabled: bool = False
    api_key: Optional[str] = Field(None, description="API key for authentication")
    
    # Rate limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = Field(100, gt=0)
    rate_limit_window_sec: int = Field(60, gt=0)
    
    # Model serving
    model_path: str
    ensemble_enabled: bool = False
    threshold: float = Field(0.5, ge=0, le=1.0)

    class Config:
        use_enum_values = False


class PCamConfig(BaseModel):
    """Root configuration for entire PCam system."""
    
    # Version
    config_version: str = "1.0.0"
    
    # Subconfigs
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    
    # Global seed
    seed: int = Field(42, ge=0)
    
    @root_validator
    def validate_global_config(cls, values):
        """Global cross-field validation."""
        seed = values.get('seed')
        if values.get('training') and values['training'].seed != seed:
            logger.warning("Training seed differs from global seed")
        return values
    
    def to_dict_nested(self) -> Dict[str, Any]:
        """Export full config as nested dictionary."""
        return self.dict(by_alias=False, exclude_none=False)
    
    def log_summary(self):
        """Log configuration summary for audit trail."""
        logger.info(f"PCam Configuration v{self.config_version}")
        logger.info(f"  Architecture: {self.model.architecture.value}")
        logger.info(f"  Training epochs: {self.training.epochs}")
        logger.info(f"  Batch size: {self.training.batch_size}")
        logger.info(f"  Optimizer: {self.training.optimizer.type.value}")
        logger.info(f"  Loss: {self.training.loss.type.value}")
        logger.info(f"  Stain norm: {self.data.stain_normalization.enabled}")
        logger.info(f"  Ensemble: {self.ensemble.enabled}")
        logger.info(f"  Seed: {self.seed}")

    class Config:
        use_enum_values = False
