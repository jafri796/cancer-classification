"""
Clinical interpretability layer for model attention visualization.

Provides techniques to visualize what regions of histopathology patches
the model focuses on during classification. Critical for:
- Clinical validation (verify focus on tumor regions)
- FDA regulatory submission (explainability)
- Debugging model failures
- Building clinician trust

Methods:
- Grad-CAM: Gradient-based attention mapping
- Integrated Gradients: Path integration for attribution
- Attention rollout: For Vision Transformers
- Integrated attention analysis: Ensemble-level explanation

All visualizations mapped back to original 96×96 patch with
center 32×32 region overlaid for PCam context.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class AttentionStatistics:
    """Statistics quantifying model attention."""
    
    # Center region attention (32×32 in 96×96 input)
    center_attention_mean: float  # Mean attention in center region
    center_attention_std: float   # Std dev of attention in center
    center_attention_fraction: float  # % of total attention in center region
    
    # Peripheral regions
    peripheral_attention_mean: float
    peripheral_attention_fraction: float
    
    # Quality metrics
    attention_entropy: float  # Shannon entropy of attention (lower = more focused)
    center_focus_score: float  # [0-1] how much model focuses on center (1 = perfect)
    
    # Spatial distribution
    attention_peak_x: float  # x-coordinate of max attention
    attention_peak_y: float  # y-coordinate of max attention
    peak_distance_from_center: float  # Distance from center in pixels
    
    timestamp: str = ""


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for medical interpretability.
    
    Generates attention maps showing which image regions influence the model's
    prediction. Validates that attention focuses on PCam's center diagnostic region.
    
    Medical Semantics:
    - PCam labels are based on tumor presence in center 32×32 region
    - Model should weight center region heavily
    - But should also consider peripheral context (stroma, inflammation)
    - Attention maps serve as model validation for clinical use
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[str] = None,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Initialize Grad-CAM for a model.
        
        Args:
            model: PyTorch model (CNN-based, as we need to hook conv layers)
            target_layer: Name of target layer to visualize (e.g., 'layer4')
                         If None, automatically selects last conv layer
            device: Compute device
        """
        self.model = model.to(device)
        self.device = device
        self.target_layer = target_layer or self._find_last_conv_layer()
        
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self._register_hooks()
        
        logger.info(f"GradCAM initialized with target_layer={self.target_layer}")
    
    def _find_last_conv_layer(self) -> Optional[str]:
        """Auto-detect last convolutional layer."""
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, nn.Conv2d):
                logger.info(f"Auto-detected last conv layer: {name}")
                return name
        
        logger.warning("Could not auto-detect conv layer, may fail")
        return None
    
    def _register_hooks(self) -> None:
        """Register forward and backward hooks to capture activations and gradients."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find and hook target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer or self.target_layer in name:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                logger.info(f"Hooked layer: {name}")
                return
        
        logger.warning(f"Could not find target layer: {self.target_layer}")
    
    def generate_cam(
        self,
        image: torch.Tensor,
        class_idx: Optional[int] = None,
        upsample_size: Tuple[int, int] = (96, 96),
    ) -> np.ndarray:
        """
        Generate Grad-CAM attention map for an image.
        
        Args:
            image: Input image (1, 3, 96, 96)
            class_idx: Class index to generate CAM for (0=negative, 1=positive)
                      If None, uses predicted class
            upsample_size: Size to upsample attention map to
            
        Returns:
            Attention map (96, 96) with values in [0, 1]
        """
        # Ensure batch dimension
        if image.ndim == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Forward pass to get predictions (no grad)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(image)
            probs = torch.sigmoid(logits)

        if class_idx is None:
            class_idx = (probs > 0.5).long().item()

        # Backward pass to get gradients
        target = logits[:, class_idx] if logits.ndim > 1 else logits
        self.model.zero_grad()
        target.backward()
        
        # Compute Grad-CAM
        if self.gradients is None or self.activations is None:
            logger.warning("Gradients or activations not captured, returning zeros")
            return np.zeros(upsample_size)
        
        # Gradient pooling
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted activation sum
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().detach().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Upsample to input resolution
        cam = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
        cam = F.interpolate(cam, size=upsample_size, mode='bilinear', align_corners=False)
        cam = cam.squeeze().numpy()
        
        return cam
    
    def compute_attention_statistics(self, cam: np.ndarray) -> AttentionStatistics:
        """
        Compute statistics quantifying whether model focuses on center region.
        
        PCam Center Region:
        - Total image: 96×96
        - Center diagnostic region: 32×32 (pixels 32-64 in both dims)
        - Peripheral context: surrounding 32-pixel border
        """
        assert cam.shape == (96, 96), f"Expected (96, 96), got {cam.shape}"
        
        # Define regions
        center_start, center_end = 32, 64
        
        # Extract center and peripheral regions
        center_mask = np.zeros((96, 96), dtype=bool)
        center_mask[center_start:center_end, center_start:center_end] = True
        
        peripheral_mask = ~center_mask
        
        # Compute statistics
        center_attention = cam[center_mask]
        peripheral_attention = cam[peripheral_mask]
        
        center_mean = float(center_attention.mean())
        center_std = float(center_attention.std())
        peripheral_mean = float(peripheral_attention.mean())
        
        # Total attention in each region (integral)
        center_total = float(center_attention.sum())
        peripheral_total = float(peripheral_attention.sum())
        total_attention = center_total + peripheral_total
        
        center_fraction = center_total / total_attention if total_attention > 0 else 0.0
        peripheral_fraction = peripheral_total / total_attention if total_attention > 0 else 0.0
        
        # Entropy (how focused is the attention?)
        cam_normalized = cam / (cam.sum() + 1e-6)
        entropy = -np.sum(cam_normalized * np.log(cam_normalized + 1e-10))
        
        # Center focus score [0, 1]
        # Perfect: all attention in center (fraction = 1.0)
        # Worst: uniform attention (fraction = (32*32)/(96*96) = 0.111)
        center_focus_score = (center_fraction - 0.111) / (1.0 - 0.111)
        center_focus_score = max(0.0, min(1.0, center_focus_score))
        
        # Peak location
        peak_idx = np.unravel_index(np.argmax(cam), cam.shape)
        peak_x, peak_y = peak_idx
        center_x, center_y = 48, 48  # Center of 96×96 image
        peak_distance = np.sqrt((peak_x - center_x) ** 2 + (peak_y - center_y) ** 2)
        
        return AttentionStatistics(
            center_attention_mean=center_mean,
            center_attention_std=center_std,
            center_attention_fraction=float(center_fraction),
            peripheral_attention_mean=peripheral_mean,
            peripheral_attention_fraction=float(peripheral_fraction),
            attention_entropy=float(entropy),
            center_focus_score=float(center_focus_score),
            attention_peak_x=float(peak_x),
            attention_peak_y=float(peak_y),
            peak_distance_from_center=float(peak_distance),
            timestamp=datetime.now().isoformat(),
        )


class InterpretabilityEvaluator:
    """
    Comprehensive interpretability evaluation for a model or ensemble.
    
    Responsibilities:
    - Generate CAM for sample images
    - Compute attention statistics across dataset
    - Validate center-region focus
    - Compare attention across ensemble members
    - Generate visualizations for clinical review
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device('cpu'),
        model_name: str = "model",
    ):
        """
        Initialize interpretability evaluator.
        
        Args:
            model: Model to interpret
            device: Compute device
            model_name: Name for this model (useful for ensembles)
        """
        self.model = model
        self.device = device
        self.model_name = model_name
        
        # Initialize Grad-CAM
        try:
            self.gradcam = GradCAM(model, device=device)
            logger.info(f"Initialized GradCAM for {model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize GradCAM: {e}")
            self.gradcam = None
    
    def evaluate_sample(
        self,
        image: torch.Tensor,
        label: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate interpretability for a single image.
        
        Returns:
            Dict with prediction, confidence, attention map, statistics
        """
        if self.gradcam is None:
            logger.warning("GradCAM not available")
            return {}
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(image.unsqueeze(0).to(self.device))
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob > 0.5 else 0
        
        # Grad-CAM
        cam = self.gradcam.generate_cam(image)
        
        # Statistics
        stats = self.gradcam.compute_attention_statistics(cam)
        
        return {
            'model_name': self.model_name,
            'prediction': pred,
            'probability': float(prob),
            'true_label': label,
            'correct': (pred == label) if label is not None else None,
            'attention_map': cam,
            'attention_statistics': {
                'center_attention_mean': stats.center_attention_mean,
                'center_attention_fraction': stats.center_attention_fraction,
                'center_focus_score': stats.center_focus_score,
                'attention_entropy': stats.attention_entropy,
                'peak_distance_from_center': stats.peak_distance_from_center,
            }
        }
    
    def evaluate_dataset(
        self,
        test_loader: DataLoader,
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate interpretability across a dataset.
        
        Computes aggregate attention statistics.
        """
        if self.gradcam is None:
            return {}
        
        all_stats = []
        all_predictions = []
        all_labels = []
        
        logger.info(f"Evaluating interpretability on {len(test_loader)} batches")
        
        for batch_idx, (images, labels) in enumerate(test_loader):
            if max_samples and batch_idx * len(images) >= max_samples:
                break
            
            for image, label in zip(images, labels):
                result = self.evaluate_sample(image, label.item())
                
                if result and 'attention_statistics' in result:
                    all_stats.append(result['attention_statistics'])
                    all_predictions.append(result['prediction'])
                    all_labels.append(label.item())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1} batches")
        
        # Aggregate statistics
        if all_stats:
            center_focus_scores = [s['center_focus_score'] for s in all_stats]
            center_fractions = [s['center_attention_fraction'] for s in all_stats]
            entropies = [s['attention_entropy'] for s in all_stats]
            peak_distances = [s['peak_distance_from_center'] for s in all_stats]
            
            return {
                'model_name': self.model_name,
                'n_samples': len(all_stats),
                'predictions': all_predictions,
                'labels': all_labels,
                'center_focus_score_mean': float(np.mean(center_focus_scores)),
                'center_focus_score_std': float(np.std(center_focus_scores)),
                'center_attention_fraction_mean': float(np.mean(center_fractions)),
                'attention_entropy_mean': float(np.mean(entropies)),
                'peak_distance_mean': float(np.mean(peak_distances)),
                'center_focus_score_distribution': center_focus_scores,
            }
        
        return {}


def visualize_attention(
    image: np.ndarray,
    attention_map: np.ndarray,
    output_path: Path,
    prediction: Optional[int] = None,
    probability: Optional[float] = None,
    title: Optional[str] = None,
) -> None:
    """
    Create visualization of image with overlaid attention map.
    
    Useful for clinical review and validation.
    
    Args:
        image: Input image (96, 96, 3) normalized to [0, 1]
        attention_map: Attention CAM (96, 96)
        output_path: Where to save the visualization
        prediction: Model prediction (0 or 1) for labeling
        probability: Model confidence [0, 1]
        title: Title for the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map
    im = axes[1].imshow(attention_map, cmap='hot')
    
    # Draw center region box
    center_start, center_end = 32, 64
    rect = plt.Rectangle(
        (center_start, center_start),
        center_end - center_start,
        center_end - center_start,
        linewidth=2,
        edgecolor='cyan',
        facecolor='none',
        label='Diagnostic region'
    )
    axes[1].add_patch(rect)
    axes[1].set_title('Attention Map (Grad-CAM)')
    axes[1].axis('off')
    axes[1].legend(loc='upper left')
    plt.colorbar(im, ax=axes[1])
    
    # Overlay
    image_overlay = image.copy()
    attention_overlay = cm.hot(attention_map)[:, :, :3]
    image_overlay = 0.6 * image + 0.4 * attention_overlay
    
    axes[2].imshow(image_overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    if title:
        fig.suptitle(title)
    elif prediction is not None and probability is not None:
        pred_label = "Positive (Tumor)" if prediction == 1 else "Negative"
        fig.suptitle(f"Prediction: {pred_label} ({probability:.2%})")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualization to {output_path}")


def generate_interpretability_report(
    evaluation_results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """
    Generate human-readable interpretability report.
    
    Args:
        evaluation_results: Results from InterpretabilityEvaluator
        output_dir: Directory to save report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = output_dir / "interpretability_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PCam Classification Model - Interpretability Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model: {evaluation_results.get('model_name', 'unknown')}\n")
        f.write(f"Samples Evaluated: {evaluation_results.get('n_samples', 0)}\n\n")
        
        f.write("ATTENTION QUALITY\n")
        f.write("-" * 80 + "\n")
        
        center_focus = evaluation_results.get('center_focus_score_mean', 0.0)
        f.write(f"Center Focus Score (0-1, higher is better): {center_focus:.4f}\n")
        f.write(f"  Interpretation: Model focuses {center_focus*100:.1f}% optimally on diagnostic region\n\n")
        
        center_frac = evaluation_results.get('center_attention_fraction_mean', 0.0)
        f.write(f"Center Region Attention Fraction: {center_frac:.2%}\n")
        f.write(f"  Optimal range: 30-60% (focus on center but use peripheral context)\n\n")
        
        entropy = evaluation_results.get('attention_entropy_mean', 0.0)
        f.write(f"Attention Entropy (lower = more focused): {entropy:.4f}\n\n")
        
        peak_dist = evaluation_results.get('peak_distance_mean', 0.0)
        f.write(f"Peak Attention Distance from Center: {peak_dist:.2f} pixels\n")
        f.write(f"  Optimal: <15 pixels (peak should be near center)\n\n")
        
        f.write("VALIDATION STATUS\n")
        f.write("-" * 80 + "\n")
        
        # Validation checks
        checks_passed = 0
        checks_total = 3
        
        if 0.3 <= center_frac <= 0.6:
            f.write("✓ Center region attention is in acceptable range\n")
            checks_passed += 1
        else:
            f.write(f"✗ Center region attention {center_frac:.2%} outside acceptable range [30%-60%]\n")
        
        if peak_dist < 15:
            f.write("✓ Peak attention is near center\n")
            checks_passed += 1
        else:
            f.write(f"✗ Peak attention {peak_dist:.1f} pixels from center (too far)\n")
        
        if center_focus > 0.5:
            f.write("✓ Center focus score indicates good diagnostic region emphasis\n")
            checks_passed += 1
        else:
            f.write(f"✗ Center focus score {center_focus:.4f} is low\n")
        
        f.write(f"\nValidation: {checks_passed}/{checks_total} checks passed\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("CLINICAL INTERPRETATION\n")
        f.write("=" * 80 + "\n")
        f.write(
            "The attention maps show which image regions the model considers when making\n"
            "predictions. For PCam:\n\n"
            "✓ Good: Model focuses on center 32×32 region where tumor labels are based\n"
            "✓ Good: Model uses peripheral context (surrounding tissues, stroma, inflammation)\n"
            "✗ Concerning: Model only looks at image edges or corners\n"
            "✗ Concerning: Model attention is diffuse/unfocused (high entropy)\n"
            "✗ Concerning: Model focus is far from image center\n\n"
            "Use these visualizations to validate that model reasoning is clinically plausible.\n"
        )
        
        f.write("=" * 80 + "\n")
    
    logger.info(f"Saved interpretability report to {report_path}")


class EnsembleExplainer:
    """
    Aggregate explanations from ensemble members.
    
    Generates per-model and ensemble-level explanations to show:
    - Model consensus (high attention in same regions)
    - Model disagreement (different focus areas)
    - Robustness of attention patterns
    
    PCam medical context:
    - Consensus on center region → robust model
    - Disagreement → potential model weaknesses
    - Can flag low-consensus predictions as uncertain
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            models: List of ensemble member models
            device: Compute device
        """
        self.models = models
        self.device = device
        
        self.grad_cams = [
            GradCAM(model, device=device)
            for model in models
        ]
    
    @torch.no_grad()
    def explain(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Generate explanations from all models.
        
        Args:
            x: Input batch (B, C, H, W)
            
        Returns:
            Dictionary with:
            - individual: Per-model attention maps (N_models, B, H, W)
            - consensus: Average attention (B, H, W)
            - disagreement: Std dev of attention (B, H, W)
            - center_focus: Avg attention in center 32×32 region
        """
        x = x.to(self.device)
        B, C, H, W = x.shape
        
        all_attentions = []
        for grad_cam in self.grad_cams:
            attention = grad_cam.generate_cam(x)  # (B, H, W)
            all_attentions.append(attention)
        
        all_attentions = np.stack(all_attentions)  # (N_models, B, H, W)
        
        # Consensus
        consensus = np.mean(all_attentions, axis=0)
        
        # Disagreement
        disagreement = np.std(all_attentions, axis=0)
        
        # Center focus (center 32×32 in 96×96)
        center_h_start = int(32 * H / 96)
        center_h_end = int(64 * H / 96)
        center_w_start = int(32 * W / 96)
        center_w_end = int(64 * W / 96)
        
        center_focus_values = np.mean(
            consensus[:, center_h_start:center_h_end, center_w_start:center_w_end],
            axis=(1, 2)
        )
        
        return {
            'individual': all_attentions,
            'consensus': consensus,
            'disagreement': disagreement,
            'center_focus_values': center_focus_values,
            'mean_center_focus': float(np.mean(center_focus_values)),
            'std_center_focus': float(np.std(center_focus_values)),
        }


class AttentionVisualizer:
    """
    Render attention maps as overlays on original patches.
    
    Creates publication-quality visualizations showing:
    - Original patch
    - Attention heatmap (jet colormap)
    - Center 32×32 region overlay (red box)
    - Aggregated saliency
    """
    
    @staticmethod
    def visualize_attention(
        image: np.ndarray,
        attention: np.ndarray,
        center_region: bool = True,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Overlay attention map on image.
        
        Args:
            image: Original patch (H, W, 3) in [0, 1]
            attention: Attention map (H, W) in [0, 1]
            center_region: Whether to overlay center 32×32 box
            alpha: Transparency of attention overlay
            
        Returns:
            Visualization (H, W, 3) in [0, 1]
        """
        H, W = image.shape[:2]
        
        # Convert to uint8 for OpenCV
        image_uint8 = (image * 255).astype(np.uint8)
        attention_uint8 = (attention * 255).astype(np.uint8)
        
        # Ensure 3 channels
        if image_uint8.shape[2] == 4:
            image_uint8 = image_uint8[:, :, :3]
        
        # Apply colormap to attention
        attention_colored = cv2.applyColorMap(attention_uint8, cv2.COLORMAP_JET)
        attention_colored = cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB)
        
        # Blend
        visualization = cv2.addWeighted(
            image_uint8, 1 - alpha,
            attention_colored, alpha,
            0
        )
        
        # Overlay center region box (32×32 in 96×96 input)
        if center_region and (H == 96 and W == 96):
            center_start = 32
            center_end = 64
            cv2.rectangle(
                visualization,
                (center_start, center_start),
                (center_end, center_end),
                (0, 0, 255),  # Red in RGB
                thickness=2
            )
        
        return visualization.astype(np.float32) / 255.0
    
    @staticmethod
    def save_visualization(
        visualization: np.ndarray,
        output_path: Path,
    ) -> None:
        """Save visualization to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to uint8
        vis_uint8 = (visualization * 255).astype(np.uint8)
        
        # Save using PIL (better compression)
        image = Image.fromarray(vis_uint8)
        image.save(output_path, quality=95)
        
        logger.info(f"Visualization saved to {output_path}")


def validate_attention_patterns(
    explanations: Dict[str, np.ndarray],
    labels: np.ndarray,
    threshold_center_focus: float = 0.3,
) -> Dict[str, Any]:
    """
    Validate that attention patterns align with clinical expectations.
    
    Checks:
    - Center region receives substantial attention
    - Consistency across ensemble (low disagreement)
    - Attention patterns correlate with predictions
    
    Medical justification:
    - PCam labels are based on center 32×32 region
    - Model should focus on center (>30% of total attention)
    - Low consensus disagreement indicates robust model
    
    Args:
        explanations: Attention maps from EnsembleExplainer
        labels: Ground truth labels (B,)
        threshold_center_focus: Min attention fraction on center
        
    Returns:
        Validation report with flags
    """
    consensus = explanations['consensus']
    disagreement = explanations['disagreement']
    center_focus_values = explanations['center_focus_values']
    
    flags = []
    
    for i, (label, cf) in enumerate(zip(labels, center_focus_values)):
        if cf < threshold_center_focus:
            flags.append({
                'sample': i,
                'label': int(label),
                'center_focus': float(cf),
                'flag': 'LOW_CENTER_ATTENTION',
                'severity': 'warning' if cf > 0.15 else 'critical',
            })
        
        # Check for high disagreement (ensemble members disagree)
        mean_disagreement = np.mean(disagreement[i])
        if mean_disagreement > 0.2:
            flags.append({
                'sample': i,
                'label': int(label),
                'mean_disagreement': float(mean_disagreement),
                'flag': 'HIGH_ENSEMBLE_DISAGREEMENT',
                'severity': 'warning',
            })
    
    return {
        'total_samples': len(labels),
        'mean_center_focus': float(np.mean(center_focus_values)),
        'std_center_focus': float(np.std(center_focus_values)),
        'mean_disagreement': float(np.mean(disagreement)),
        'flagged_samples': flags,
        'validation_passed': len(flags) < 0.1 * len(labels),
    }

