# Quick Start: Production System Guide

## Overview

This is a comprehensive PCam cancer classification system ready for production deployment with:
- ✅ Production-grade callbacks system
- ✅ Comprehensive held-out test evaluation
- ✅ Clinical interpretability (Grad-CAM)
- ✅ OOD/anomaly detection
- ✅ FDA regulatory compliance checklist

## 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 2. Training with Production Callbacks

### Basic Training

```python
from src.training.trainer import Trainer
from src.training.callbacks import get_default_callbacks
from pathlib import Path

# Initialize trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    device=device,
)

# Callbacks automatically included:
# - ModelCheckpoint: Best model + periodic saves
# - EarlyStopping: Stop if val_auc plateaus
# - MetricsLogger: Console + MLflow logging
# - ClinicalMetricsLogger: Sensitivity/specificity monitoring
# - LearningRateMonitor: Learning rate tracking

# Train
history = trainer.fit(epochs=100)
```

### Key Callback Features

- **ModelCheckpoint**: Saves model when val_auc improves
  - Best model saved to `checkpoint_dir/best_model.pt`
  - Metadata includes: epoch, metrics, config

- **EarlyStopping**: Stops training with patience
  - Default: patience=10 (stops if no improvement for 10 epochs)
  - Supports clinical constraints (e.g., don't stop if sensitivity < 95%)

- **ClinicalMetricsLogger**: Alerts if metrics below targets
  - Sensitivity target: ≥95%
  - Specificity target: ≥90%

## 3. Comprehensive Test Evaluation

```python
from src.validation.held_out_test_evaluation import HeldOutTestEvaluator

# Create evaluator
evaluator = HeldOutTestEvaluator(
    model_path='checkpoints/best_model.pt',
    test_loader=test_loader,
    device=device,
    model_class=MyModelClass,
    bootstrap_samples=1000,
    ci_level=0.95,
)

# Run evaluation
results = evaluator.evaluate()

# Generate visualizations
evaluator.plot_roc_curve('test_roc_curve.png')
evaluator.plot_calibration_curve('test_calibration_curve.png')
evaluator.plot_confusion_matrix('test_confusion_matrix.png')

# Save results
evaluator.save_report('test_evaluation_report.json')
```

### Output

- **Metrics**:
  - AUC-ROC (primary metric)
  - Sensitivity, Specificity, PPV, NPV
  - 95% bootstrap confidence intervals

- **Visualizations**:
  - ROC curve with AUC value
  - Calibration curve (ECE metric)
  - Confusion matrix

- **Report**: JSON with complete metrics and operating points

## 4. Clinical Interpretability

```python
from src.inference.interpretability import (
    EnsembleExplainer, AttentionVisualizer,
    validate_attention_patterns
)

# Create ensemble explainer
explainer = EnsembleExplainer(models=[model1, model2, model3], device=device)

# Generate explanations
explanations = explainer.explain(images)
# Returns: individual, consensus, disagreement, center_focus

# Validate attention patterns
validation = validate_attention_patterns(
    explanations,
    labels,
    threshold_center_focus=0.3
)

if validation['validation_passed']:
    print("✓ Attention patterns valid (center-region focus confirmed)")
else:
    print(f"⚠️ {len(validation['flagged_samples'])} flagged samples")

# Visualize attention
visualizer = AttentionVisualizer()
attention_viz = visualizer.visualize_attention(
    image=image_array,
    attention=attention_map,
    center_region=True,  # Show center 32×32 box
)
visualizer.save_visualization(attention_viz, Path('attention.png'))
```

### Interpretation Guide

- **Center-Region Focus (30-60%)**:
  - ✓ 30-60%: Model focuses on diagnostic region with context
  - ✗ <30%: Model ignores diagnostic center (failure mode)
  - ✗ >60%: Model ignores helpful context (suboptimal)

- **Ensemble Disagreement**:
  - ✓ Low (<0.2): Models agree (robust)
  - ✗ High (>0.2): Models disagree (uncertain)

## 5. OOD Detection

```python
from src.inference.anomaly_detection import CompositeOODDetector

# Create detector (combines Mahalanobis + Entropy + Isolation Forest)
detector = CompositeOODDetector(
    model=model,
    device=device,
    use_mahalanobis=True,
    use_entropy=True,
    use_isolation_forest=True,
)

# Fit on training/validation set
detector.fit(val_loader)

# Detect OOD samples
results = detector.detect(new_images)
# Returns: is_ood, ensemble_score, individual_scores, voting_count

# Usage during inference
for i, is_ood in enumerate(results['is_ood']):
    if is_ood:
        print(f"Sample {i}: Flagged for manual review (OOD score: {results['ensemble_score'][i]:.4f})")
    else:
        print(f"Sample {i}: Normal (confidence: {results['ensemble_score'][i]:.4f})")

# Save detector for deployment
detector.save('ood_detector.pkl')
```

### OOD Voting Strategy

- **Flag as OOD if**: ≥2 of 3 methods agree
- **Methods**:
  - Mahalanobis: Statistical distance (detects feature drift)
  - Entropy: Prediction uncertainty (detects confidence drops)
  - Isolation Forest: Anomaly score (detects rare patterns)

## 6. Production Inference

```python
from src.inference.ensemble_predictor import EnsemblePredictor
from src.inference.anomaly_detection import CompositeOODDetector, ModelDeploymentMonitor

# Load ensemble for inference
predictor = EnsemblePredictor(
    model_paths=['best_model_1.pt', 'best_model_2.pt', 'best_model_3.pt'],
    device=device,
    use_tta=True,  # Test-time augmentation
)

# Load OOD detector
detector = CompositeOODDetector(model=model, device=device)
detector.load('ood_detector.pkl')

# Setup monitoring
monitor = ModelDeploymentMonitor(log_dir=Path('deployment_logs'))

# Inference
predictions = predictor.predict(images, return_uncertainty=True)
ood_results = detector.detect(images)

# Log predictions for monitoring
for i in range(len(images)):
    monitor.log_prediction(
        image_id=f'image_{i}',
        prediction=predictions['predictions'][i],
        confidence=predictions['probabilities'][i],
        is_ood=ood_results['is_ood'][i],
        ood_score=ood_results['ensemble_score'][i],
        metadata={'sample_type': 'validation'},
    )

# Generate monitoring report
report = monitor.generate_report()
print(f"OOD detection rate: {report['ood_detection_rate']:.2%}")
for alert in report['alerts']:
    print(f"⚠️ {alert}")
```

## 7. Regulatory Compliance

### Checklist

Use **[DEPLOYMENT.md](DEPLOYMENT.md)** for comprehensive FDA 510(k) compliance:

1. ✅ **Dataset & Data Integrity** (20 items)
2. ✅ **Model Architecture** (10 items)
3. ✅ **Training Process** (12 items)
4. ✅ **Evaluation & Metrics** (15 items)
5. ✅ **Clinical Validation** (12 items)
6. ✅ **Interpretability** (8 items)
7. ✅ **Regulatory Compliance** (10 items)
8. ✅ **Quality Management** (8 items)
9. ✅ **Inference & Deployment** (8 items)
10. ✅ **Data Governance** (8 items)
11. ✅ **Testing** (12 items)
12. ✅ **Documentation** (8 items)

### Key Metrics (Fill Before Deployment)

```
Primary Metric:
  AUC-ROC: ____ (95% CI: [____, ____])

Clinical Performance (optimal threshold = ____):
  Sensitivity: ____ % (95% CI: [____, ____])
  Specificity: ____ % (95% CI: [____, ____])

Model Calibration:
  ECE: ____
  Well-calibrated: YES/NO

Attention Validation:
  Center-region focus: ____ %
  Ensemble agreement: YES/NO

OOD Detection:
  OOD rate (on val set): ____
  Detection accuracy: ____
```

## 8. Complete Example Workflow

See [workflow_production_demo.py](workflow_production_demo.py) for complete end-to-end example:

```bash
python workflow_production_demo.py
```

This demonstrates:
1. Setup with reproducibility
2. Training with callbacks
3. Test evaluation with metrics
4. Interpretability validation
5. OOD detection setup
6. Regulatory documentation
7. Deployment inference

## Key Files

| File | Purpose |
|------|---------|
| [src/training/callbacks.py](src/training/callbacks.py) | Formalized callback system |
| [src/training/trainer.py](src/training/trainer.py) | Training orchestrator (integrated with callbacks) |
| [src/validation/held_out_test_evaluation.py](src/validation/held_out_test_evaluation.py) | Comprehensive test evaluation |
| [src/inference/interpretability.py](src/inference/interpretability.py) | Grad-CAM + attention validation |
| [src/inference/anomaly_detection.py](src/inference/anomaly_detection.py) | OOD detection |
| [DEPLOYMENT.md](DEPLOYMENT.md) | FDA 510(k) regulatory checklist |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Complete implementation guide |
| [workflow_production_demo.py](workflow_production_demo.py) | End-to-end workflow example |

## Performance Targets

| Metric | Target | Status |
|--------|--------|--------|
| Test AUC-ROC | ≥0.96 | TBD |
| Test Sensitivity | ≥95% | TBD |
| Test Specificity | ≥90% | TBD |
| Inference latency (single) | <100 ms | TBD |
| Inference latency (batch-32) | <200 ms | TBD |
| Model calibration (ECE) | <0.1 | TBD |
| Attention center-focus | 30-60% | TBD |
| Ensemble agreement | Low disagreement | TBD |

## Troubleshooting

### Issue: Training diverges or loss is NaN

**Solution**:
- Check gradient norm in logs (should be < 10)
- Reduce learning rate (divide by 2-10)
- Check for data leakage (stain normalizer fit set)
- Verify data loading (check for NaN/Inf values)

### Issue: Model overfits (high train AUC, low val AUC)

**Solution**:
- Increase dropout
- Add early stopping patience
- Reduce model capacity
- Increase augmentation strength
- Check for class imbalance handling

### Issue: OOD detection rate too high

**Solution**:
- Check if training set is representative
- Adjust OOD voting threshold (currently ≥2 of 3)
- Verify normalizer fit on correct set
- Check for distribution shift in validation set

### Issue: Attention not focused on center

**Solution**:
- Verify spatial attention module enabled
- Check center region defined correctly (32×32 in 96×96)
- Visualize Grad-CAM to see what model learns
- Consider training longer (may take more epochs)

## Support

- **Technical Questions**: Check [README.md](README.md) and inline documentation
- **Regulatory Questions**: See [DEPLOYMENT.md](DEPLOYMENT.md)
- **Clinical Questions**: Consult pathology advisor

---

**System Version**: 1.0  
**Last Updated**: February 2, 2026  
**Status**: Production-Ready (pending test set validation)
