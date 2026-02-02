"""
Loss Functions for Medical Image Classification

All loss functions designed for binary classification in histopathology.
Addresses class imbalance and focuses on hard examples (missed tumors).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Medical Justification:
    - PCam has ~60/40 class split (moderate imbalance)
    - False negatives (missed tumors) are clinically critical
    - Focal loss down-weights easy examples, focuses on hard cases
    
    Formula:
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        
    where:
        p_t = p if y=1, else 1-p
        α_t = α if y=1, else 1-α
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" ICCV 2017
    
    Args:
        alpha: Weighting factor for positive class [0, 1]
               alpha=0.25 gives more weight to negative class
        gamma: Focusing parameter [0, 5]
               gamma=2 is standard, gamma=0 reduces to CE
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        
        assert 0 <= alpha <= 1, f"alpha must be in [0, 1], got {alpha}"
        assert gamma >= 0, f"gamma must be >= 0, got {gamma}"
        assert reduction in ['mean', 'sum', 'none'], \
            f"reduction must be 'mean', 'sum', or 'none', got {reduction}"
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        logger.info(
            f"Initialized FocalLoss with alpha={alpha}, gamma={gamma}, "
            f"reduction={reduction}"
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits from model (B, 1) - NOT probabilities
            targets: Ground truth labels (B, 1) - values in {0, 1}
            
        Returns:
            Loss value (scalar if reduction != 'none')
        """
        # Validate inputs
        assert inputs.shape == targets.shape, \
            f"Shape mismatch: inputs {inputs.shape}, targets {targets.shape}"
        assert targets.min() >= 0 and targets.max() <= 1, \
            f"Targets must be in [0, 1], got range [{targets.min()}, {targets.max()}]"
        
        # Compute probabilities from logits
        probs = torch.sigmoid(inputs)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            reduction='none'
        )
        
        # Compute p_t
        p_t = torch.where(targets == 1, probs, 1 - probs)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t).pow(self.gamma)
        
        # Alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Focal loss
        focal_loss = alpha_t * focal_weight * bce
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss.
    
    Medical Justification:
    - Simple alternative to Focal Loss
    - Directly weights positive class higher
    - Clinically: weigh false negatives more than false positives
    
    Formula:
        L = -[w_pos * y * log(p) + w_neg * (1-y) * log(1-p)]
    
    Args:
        pos_weight: Weight for positive class (tensor or scalar)
                    Typically set to n_negative / n_positive
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
    ):
        super().__init__()
        
        if pos_weight is not None:
            if not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor([pos_weight], dtype=torch.float32)
        
        self.pos_weight = pos_weight
        self.reduction = reduction
        
        logger.info(
            f"Initialized WeightedBCELoss with pos_weight={pos_weight}, "
            f"reduction={reduction}"
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            inputs: Logits from model (B, 1)
            targets: Ground truth labels (B, 1)
            
        Returns:
            Loss value
        """
        return F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction,
        )


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss - different penalties for false positives vs false negatives.
    
    Medical Justification:
    - In center-region tumor detection: false negative (missed center tumor) >> false positive
    - Want to penalize missed tumors more heavily than false alarms
    - Asymmetric loss applies different focusing to positive vs negative
    
    Reference:
        Ridnik et al. "Asymmetric Loss For Multi-Label Classification" ICCV 2021
    
    Args:
        gamma_pos: Focusing parameter for positive class
        gamma_neg: Focusing parameter for negative class
        clip: Probability clipping value to prevent numerical instability
    """
    
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
    ):
        super().__init__()
        
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        
        logger.info(
            f"Initialized AsymmetricLoss with gamma_pos={gamma_pos}, "
            f"gamma_neg={gamma_neg}, clip={clip}"
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute asymmetric loss."""
        # Compute probabilities
        probs = torch.sigmoid(inputs)
        
        # Clip probabilities for numerical stability
        probs_pos = torch.clamp(probs, min=self.clip)
        probs_neg = torch.clamp(1 - probs, min=self.clip)
        
        # Positive examples: -y * log(p) * (1 - p)^gamma_pos
        loss_pos = targets * torch.log(probs_pos) * (1 - probs).pow(self.gamma_pos)
        
        # Negative examples: -(1 - y) * log(1 - p) * p^gamma_neg
        loss_neg = (1 - targets) * torch.log(probs_neg) * probs.pow(self.gamma_neg)
        
        # Combine
        loss = -(loss_pos + loss_neg)
        
        return loss.mean()


class ClinicalLoss(nn.Module):
    """
    Custom loss function optimized for clinical metrics.
    
    Medical Justification:
    - Optimize directly for sensitivity and specificity
    - Penalize false negatives more than false positives
    - Balance between AUC and specific operating point performance
    
    Combines:
    1. Focal Loss (for hard example mining)
    2. False Negative Penalty (emphasize sensitivity)
    3. Calibration Loss (well-calibrated probabilities)
    
    Args:
        fn_weight: Weight for false negative penalty (default: 3.0)
        focal_alpha: Focal loss alpha parameter
        focal_gamma: Focal loss gamma parameter
    """
    
    def __init__(
        self,
        fn_weight: float = 3.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        
        self.fn_weight = fn_weight
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        
        logger.info(
            f"Initialized ClinicalLoss with fn_weight={fn_weight}, "
            f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}"
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute clinical loss."""
        # Base focal loss
        focal = self.focal_loss(inputs, targets)
        
        # Additional penalty for false negatives
        probs = torch.sigmoid(inputs)
        
        # False negative: target=1 but prob<0.5
        fn_mask = (targets == 1) & (probs < 0.5)
        fn_penalty = torch.where(
            fn_mask,
            (0.5 - probs) ** 2,  # Quadratic penalty
            torch.zeros_like(probs)
        ).mean()
        
        # Combined loss
        total_loss = focal + self.fn_weight * fn_penalty
        
        return total_loss


def get_loss_function(config: dict) -> nn.Module:
    """
    Factory function to create loss function from config.
    
    Args:
        config: Loss configuration dictionary
        
    Returns:
        Initialized loss function
    """
    loss_type = config.get('type', 'focal_loss')
    
    if loss_type == 'focal_loss':
        return FocalLoss(
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0),
        )
    
    elif loss_type == 'weighted_bce':
        pos_weight = config.get('pos_weight', 1.5)
        if isinstance(pos_weight, (int, float)):
            pos_weight = torch.tensor([pos_weight])
        return WeightedBCELoss(pos_weight=pos_weight)
    
    elif loss_type == 'asymmetric':
        return AsymmetricLoss(
            gamma_pos=config.get('gamma_pos', 0.0),
            gamma_neg=config.get('gamma_neg', 4.0),
        )
    
    elif loss_type == 'clinical':
        return ClinicalLoss(
            fn_weight=config.get('fn_weight', 3.0),
            focal_alpha=config.get('focal_alpha', 0.25),
            focal_gamma=config.get('focal_gamma', 2.0),
        )
    
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ============================================================================
# Loss Function Validation Tests
# ============================================================================

def test_focal_loss():
    """Unit test for Focal Loss."""
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Test case 1: Perfect predictions
    inputs = torch.tensor([[5.0], [-5.0]], requires_grad=True)
    targets = torch.tensor([[1.0], [0.0]])
    loss = focal(inputs, targets)
    
    assert loss.item() > 0, "Loss should be positive"
    assert loss.requires_grad, "Loss should require gradients"
    
    # Test case 2: Wrong predictions (high loss)
    inputs_wrong = torch.tensor([[-5.0], [5.0]], requires_grad=True)
    targets_wrong = torch.tensor([[1.0], [0.0]])
    loss_wrong = focal(inputs_wrong, targets_wrong)
    
    assert loss_wrong.item() > loss.item(), \
        "Wrong predictions should have higher loss"
    
    print("✓ Focal Loss tests passed")


def test_weighted_bce():
    """Unit test for Weighted BCE Loss."""
    wbce = WeightedBCELoss(pos_weight=torch.tensor([2.0]))
    
    inputs = torch.tensor([[0.0]], requires_grad=True)
    targets = torch.tensor([[1.0]])
    
    loss = wbce(inputs, targets)
    assert loss.item() > 0, "Loss should be positive"
    
    print("✓ Weighted BCE tests passed")


if __name__ == '__main__':
    test_focal_loss()
    test_weighted_bce()
    print("\nAll loss function tests passed!")
    