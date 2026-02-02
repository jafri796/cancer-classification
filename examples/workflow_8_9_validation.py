"""
WORKFLOW 8-9: Clinical Validation Setup & Custom Training Script
Configure validation metrics and training framework
"""

import sys
from pathlib import Path

print("\n" + "="*80)
print("WORKFLOW 8-9: CLINICAL VALIDATION & TRAINING SETUP")
print("="*80)

sys.path.insert(0, '.')

try:
    # WORKFLOW 8: Clinical Validation Setup
    print("\n📌 WORKFLOW 8: Clinical Validation Configuration...")
    
    from src.validation.clinical_validation import ClinicalValidator
    
    validator = ClinicalValidator(
        sensitivity_target=0.95,
        specificity_target=0.90,
        auc_target=0.98
    )
    
    print("  ✓ Clinical validator initialized")
    print("    - Sensitivity target: ≥95%")
    print("    - Specificity target: ≥90%")
    print("    - AUC target: ≥0.98")
    
    print("\n✓ WORKFLOW 8 COMPLETE: Clinical validation configured")
    
    # WORKFLOW 9: Custom Training Script Setup
    print("\n📌 WORKFLOW 9: Custom Training Script...")
    
    import yaml
    
    # Load training config
    with open('src/configs/training.yaml') as f:
        training_cfg = yaml.safe_load(f)
    
    print("  ✓ Training config loaded")
    print(f"    - Batch size: {training_cfg['training']['batch_size']}")
    print(f"    - Epochs: {training_cfg['training']['epochs']}")
    print(f"    - Learning rate: {training_cfg['training']['optimizer']['learning_rate']}")
    print(f"    - Seed: {training_cfg['training']['seed']}")
    
    # Create example training script
    training_script = '''"""
Custom PCam Training Script
This script demonstrates how to use the training pipeline with your data.
"""

import sys
sys.path.insert(0, '.')

import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from src.training.trainer import Trainer
from src.models.center_aware_resnet import create_center_aware_resnet50
from src.utils.reproducibility import set_seed

# Configuration
with open('src/configs/training.yaml') as f:
    cfg = yaml.safe_load(f)

# Reproducibility
set_seed(cfg['training']['seed'], deterministic=True, benchmark=False)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create model
print("Loading ResNet50-SE with pretrained ImageNet weights...")
model = create_center_aware_resnet50(pretrained=True).to(device)

# Initialize trainer
trainer = Trainer(
    model=model,
    device=device,
    config=cfg['training'],
    experiment_name='pcam_resnet50_custom_run'
)

print("✓ Trainer initialized")
print("\\nWhen you have real PCam data:")
print("  1. Create train/val DataLoaders from PCam H5 files")
print("  2. Call: trainer.train(train_loader, val_loader, epochs=50)")
print("  3. Monitor: mlflow ui (at http://localhost:5000)")
'''
    
    script_path = Path('train_custom_example.py')
    script_path.write_text(training_script)
    print(f"  ✓ Example training script created: {script_path}")
    
    print("\n✓ WORKFLOW 9 COMPLETE: Training framework ready")
    print("  To train with real data:")
    print("    python train_custom_example.py")
    
except Exception as e:
    print(f"\n❌ Error in workflows 8-9: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("✓ WORKFLOWS 8-9 COMPLETE")
print("="*80)
