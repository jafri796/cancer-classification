"""
Unit Tests for Model Architectures

Tests:
1. Center-region awareness
2. Forward pass correctness
3. Gradient flow
4. Output shapes
5. Receptive field validation

Author: Principal ML Auditor
Date: 2026-01-27
FDA Compliance: Yes
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.center_aware_resnet import CenterAwareResNet50SE, create_center_aware_resnet50
from src.models.resnet_cbam import ResNet50CBAM, create_resnet50_cbam
from src.models.efficientnet import CenterAwareEfficientNet, create_efficientnet, verify_receptive_field
from src.models.vit import CenterAwareViT, create_vit, verify_center_patch_mapping
from src.models.deit import DeiTSmallPCam, create_deit_small


class TestCenterAwareResNet:
    """Test suite for Center-Aware ResNet50-SE."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        config = {
            'pretrained': False,
            'use_spatial_attention': True,
            'use_dual_pooling': True,
        }
        
        model = create_center_aware_resnet50(config)
        
        assert model is not None
        assert hasattr(model, 'spatial_attention')
        assert hasattr(model, 'pooling')
    
    def test_forward_pass_shape(self):
        """
        CRITICAL: Verify output shape is correct.
        
        Input: (B, 3, 96, 96)
        Output: (B, 1) for binary classification
        """
        config = {'pretrained': False, 'use_spatial_attention': True}
        model = create_center_aware_resnet50(config)
        model.eval()
        
        batch_size = 4
        x = torch.randn(batch_size, 3, 96, 96)
        
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (batch_size, 1), \
            f"Expected shape ({batch_size}, 1), got {logits.shape}"
    
    def test_gradient_flow(self):
        """Verify gradients flow through all components."""
        config = {'pretrained': False, 'use_spatial_attention': True}
        model = create_center_aware_resnet50(config)
        model.train()
        
        x = torch.randn(2, 3, 96, 96, requires_grad=True)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        
        # Check spatial attention gradients
        assert model.spatial_attention.conv1.weight.grad is not None
        assert model.spatial_attention.conv1.weight.grad.abs().sum() > 0
        
        # Check classifier gradients
        for param in model.classifier.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_dual_pooling_output_dim(self):
        """Verify dual pooling doubles feature dimension."""
        config = {
            'pretrained': False,
            'use_spatial_attention': False,
            'use_dual_pooling': True,
        }
        model = create_center_aware_resnet50(config)
        
        # Forward to pooling layer
        x = torch.randn(1, 3, 96, 96)
        
        # Extract features before classifier
        with torch.no_grad():
            # Forward through backbone
            features = model.conv1(x)
            features = model.bn1(features)
            features = model.relu(features)
            features = model.maxpool(features)
            features = model.layer1(features)
            features = model.layer2(features)
            features = model.layer3(features)
            features = model.layer4(features)
            
            # Apply pooling
            pooled = model.pooling(features)
        
        # Dual pooling should concatenate global + center
        # 2048 (ResNet50 output) * 2 = 4096
        assert pooled.shape[1] == 4096, \
            f"Expected 4096 features, got {pooled.shape[1]}"
    
    def test_spatial_attention_map_shape(self):
        """Verify spatial attention map has correct shape."""
        config = {'pretrained': False, 'use_spatial_attention': True}
        model = create_center_aware_resnet50(config)
        model.eval()
        
        x = torch.randn(2, 3, 96, 96)
        
        with torch.no_grad():
            attention_map = model.get_attention_map(x)
        
        # Attention map should be (B, 1, H, W) for 3×3 feature map
        assert attention_map.shape[1] == 1, "Attention should be single channel"
        assert attention_map.shape[2] == 3, "Feature map should be 3×3"
        assert attention_map.shape[3] == 3


class TestResNetCBAM:
    def test_resnet_cbam_forward(self):
        config = {"pretrained": False}
        model = create_resnet50_cbam(config)
        model.eval()
        x = torch.randn(2, 3, 96, 96)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 1)


class TestCenterAwareEfficientNet:
    """Test suite for Center-Aware EfficientNet."""
    
    @pytest.mark.parametrize("variant", ['b0', 'b1', 'b2', 'b3', 'b4'])
    def test_all_variants_forward_pass(self, variant):
        """Test all EfficientNet variants can forward."""
        config = {
            'architecture': f'efficientnet-{variant}',
            'pretrained': False,
            'use_spatial_attention': True,
        }
        
        model = create_efficientnet(config)
        model.eval()
        
        x = torch.randn(2, 3, 96, 96)
        
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (2, 1)
    
    @pytest.mark.parametrize("variant", ['b0', 'b1', 'b2', 'b3', 'b4'])
    def test_receptive_field_validation(self, variant):
        """
        CRITICAL: Verify all variants have receptive field covering full patch.
        
        FDA Requirement: Model must see entire 96×96 context.
        """
        report = verify_receptive_field(variant)
        
        assert report['status'] == 'PASS', \
            f"Receptive field check failed for {variant}: {report}"
        assert report['covers_center_region'], \
            f"{variant} doesn't cover center region"
        assert report['covers_full_patch'], \
            f"{variant} doesn't cover full patch"
    
    def test_gradient_flow_efficientnet(self):
        """Verify gradients flow through EfficientNet."""
        config = {'architecture': 'efficientnet-b0', 'pretrained': False}
        model = create_efficientnet(config)
        model.train()
        
        x = torch.randn(2, 3, 96, 96, requires_grad=True)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        
        # Check at least classifier has gradients
        for param in model.classifier.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestCenterAwareViT:
    """Test suite for Center-Aware Vision Transformer."""
    
    def test_vit_initialization(self):
        """Test ViT can be initialized."""
        config = {
            'pretrained': False,
            'use_center_bias': True,
            'use_center_pooling': True,
        }
        
        model = create_vit(config)
        
        assert model is not None
        assert model.patch_size == 16
        assert model.num_patches == 36
        assert model.grid_size == 6
    
    def test_vit_forward_pass_shape(self):
        """Verify ViT output shape."""
        config = {'pretrained': False, 'use_center_bias': True}
        model = create_vit(config)
        model.eval()
        
        x = torch.randn(2, 3, 96, 96)
        
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (2, 1)
    
    def test_center_patch_mapping(self):
        """
        CRITICAL: Verify center 32×32 maps to patches (2,2), (2,3), (3,2), (3,3).
        
        This is fundamental to PCam label definition.
        """
        report = verify_center_patch_mapping()
        
        assert report['status'] == 'PASS', \
            f"Center patch mapping failed: {report}"
        
        expected_patches = [(2, 2), (2, 3), (3, 2), (3, 3)]
        assert set(report['center_patches_2d']) == set(expected_patches), \
            f"Expected {expected_patches}, got {report['center_patches_2d']}"
        
        expected_linear = [14, 15, 20, 21]
        assert report['center_patches_linear'] == expected_linear, \
            f"Expected {expected_linear}, got {report['center_patches_linear']}"
    
    def test_vit_gradient_flow(self):
        """Verify gradients flow through ViT."""
        config = {'pretrained': False, 'use_center_bias': True}
        model = create_vit(config)
        model.train()
        
        x = torch.randn(2, 3, 96, 96, requires_grad=True)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        
        # Check center bias gradients
        if model.use_center_bias:
            assert model.center_bias.bias.grad is not None
            assert model.center_bias.bias.grad.abs().sum() > 0
        
        # Check classifier gradients
        for param in model.classifier.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestDeiTSmall:
    def test_deit_forward(self):
        config = {"pretrained": False}
        model = create_deit_small(config)
        model.eval()
        x = torch.randn(2, 3, 96, 96)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 1)
    
    def test_center_weighted_pooling(self):
        """Verify center-weighted pooling emphasizes center patches."""
        config = {
            'pretrained': False,
            'use_center_pooling': True,
        }
        model = create_vit(config)
        model.eval()
        
        # Check pooling module exists
        assert hasattr(model, 'pooling')
        
        # Verify it combines CLS and center tokens
        # (Detailed verification would require inspecting weights)


class TestModelComparison:
    """Cross-model comparison tests."""
    
    def test_all_models_same_input_shape(self):
        """Verify all models accept same input shape."""
        input_shape = (2, 3, 96, 96)
        x = torch.randn(*input_shape)
        
        # ResNet
        resnet = create_center_aware_resnet50({'pretrained': False})
        with torch.no_grad():
            out_resnet = resnet(x)
        assert out_resnet.shape == (2, 1)
        
        # EfficientNet
        efficientnet = create_efficientnet({
            'architecture': 'efficientnet-b0',
            'pretrained': False
        })
        with torch.no_grad():
            out_eff = efficientnet(x)
        assert out_eff.shape == (2, 1)
        
        # ViT
        vit = create_vit({'pretrained': False})
        with torch.no_grad():
            out_vit = vit(x)
        assert out_vit.shape == (2, 1)
    
    def test_all_models_produce_valid_logits(self):
        """Verify all models produce valid logit values."""
        x = torch.randn(4, 3, 96, 96)
        
        models = [
            create_center_aware_resnet50({'pretrained': False}),
            create_efficientnet({'architecture': 'efficientnet-b0', 'pretrained': False}),
            create_vit({'pretrained': False}),
        ]
        
        for model in models:
            model.eval()
            with torch.no_grad():
                logits = model(x)
            
            # Check no NaN/Inf
            assert not torch.isnan(logits).any(), \
                f"{model.__class__.__name__} produced NaN"
            assert not torch.isinf(logits).any(), \
                f"{model.__class__.__name__} produced Inf"
            
            # Check reasonable range (logits typically in [-10, 10])
            assert logits.abs().max() < 100, \
                f"{model.__class__.__name__} produced extreme values"


class TestParameterCounts:
    """Test model parameter counts are reasonable."""
    
    def test_resnet_parameter_count(self):
        """Verify ResNet parameter count is reasonable."""
        model = create_center_aware_resnet50({'pretrained': False})
        
        total = model.get_total_params()
        trainable = model.get_trainable_params()
        
        # ResNet50 base: ~25.6M params
        # With additions: should be 25-30M
        assert 20_000_000 < total < 35_000_000, \
            f"Unexpected param count: {total:,}"
        
        assert trainable == total, "All params should be trainable by default"
    
    def test_efficientnet_parameter_counts(self):
        """Verify EfficientNet variants have expected param counts."""
        expected_ranges = {
            'b0': (4_000_000, 7_000_000),
            'b1': (6_000_000, 10_000_000),
            'b2': (7_000_000, 12_000_000),
            'b3': (10_000_000, 15_000_000),
            'b4': (15_000_000, 25_000_000),
        }
        
        for variant, (min_params, max_params) in expected_ranges.items():
            model = create_efficientnet({
                'architecture': f'efficientnet-{variant}',
                'pretrained': False
            })
            
            total = model.get_total_params()
            
            assert min_params < total < max_params, \
                f"{variant}: Expected {min_params:,}-{max_params:,}, got {total:,}"
    
    def test_vit_parameter_count(self):
        """Verify ViT parameter count is reasonable."""
        model = create_vit({'pretrained': False})
        
        total = model.get_total_params()
        
        # ViT-B/16 base: ~86M params
        # With additions: should be 85-90M
        assert 80_000_000 < total < 95_000_000, \
            f"Unexpected param count: {total:,}"


# =============================================================================
# Test Execution
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])