"""
Complete workflow demonstrating production-grade system usage.

Shows how to integrate:
1. Formalized callbacks (checkpointing, early stopping, logging)
2. Held-out test evaluation (comprehensive metrics, bootstrap CI)
3. Clinical interpretability (Grad-CAM, attention validation)
4. OOD detection (Mahalanobis, entropy, ensemble voting)
5. Regulatory compliance (reproducibility, versioning, documentation)

This serves as a reference for deploying the system in production.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
import json
from datetime import datetime

from src.training.trainer import Trainer
from src.training.callbacks import (
    CallbackList, ModelCheckpoint, EarlyStopping,
    MetricsLogger, ClinicalMetricsLogger, LearningRateMonitor,
    get_default_callbacks
)
from src.validation.held_out_test_evaluation import HeldOutTestEvaluator
from src.inference.interpretability import (
    GradCAM, EnsembleExplainer, AttentionVisualizer,
    validate_attention_patterns
)
from src.inference.anomaly_detection import CompositeOODDetector
from src.inference.predictor import EnsemblePredictor
from src.utils.reproducibility import set_seed
from src.data.dataset import PCamDataset
from src.data.preprocessing import MedicalAugmentation
from src.models import (
    create_center_aware_resnet50,
    create_efficientnet,
    create_vit,
)

logger = logging.getLogger(__name__)


def setup_production_system(config_path: str) -> dict:
    """
    Complete setup for production system with all components.
    
    Args:
        config_path: Path to training config JSON
        
    Returns:
        Dictionary with trainer, callbacks, evaluator, etc.
    """
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    # Set seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed, deterministic=True, benchmark=False)
    logger.info(f"Set random seed to {seed} for reproducibility")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Create ensemble models
    models = create_ensemble_models(config, device)
    
    # For demonstration, we'll use the first model as primary
    model = models[0]
    
    # Setup trainer with callbacks
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        experiment_tracker=None,  # Optional: MLflow tracker
    )
    
    logger.info("✓ Trainer initialized with formalized callbacks")
    
    return {
        'trainer': trainer,
        'model': model,
        'models': models,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'config': config,
        'device': device,
        'seed': seed,
    }


def train_with_production_callbacks(system: dict) -> dict:
    """
    Train model with production-grade callbacks.
    
    Callbacks included:
    - ModelCheckpoint: Save best and periodic checkpoints
    - EarlyStopping: Stop if val_auc doesn't improve
    - MetricsLogger: Log to console and MLflow
    - ClinicalMetricsLogger: Monitor sensitivity/specificity
    - LearningRateMonitor: Track learning rate changes
    
    Args:
        system: Output from setup_production_system()
        
    Returns:
        Training history
    """
    trainer = system['trainer']
    config = system['config']
    
    logger.info("=" * 80)
    logger.info("STARTING TRAINING WITH PRODUCTION CALLBACKS")
    logger.info("=" * 80)
    
    # Training loop runs with callbacks
    history = trainer.fit(
        epochs=config.get('epochs', 100),
        val_frequency=config.get('val_frequency', 1),
    )
    
    logger.info("✓ Training completed")
    logger.info(f"  Best val_auc: {max(m['auc'] for m in history['val']):.6f}")
    logger.info(f"  Best val_sensitivity: {max(m['sensitivity'] for m in history['val']):.6f}")
    logger.info(f"  Best val_specificity: {max(m['specificity'] for m in history['val']):.6f}")
    
    return history


def evaluate_on_held_out_test(system: dict) -> dict:
    """
    Comprehensive evaluation on held-out test set.
    
    Computes:
    - Primary metric: AUC-ROC
    - Clinical metrics: Sensitivity, specificity at optimal thresholds
    - Bootstrap confidence intervals
    - Failure mode analysis
    - Calibration metrics
    
    Args:
        system: Output from setup_production_system()
        
    Returns:
        Test evaluation results
    """
    logger.info("=" * 80)
    logger.info("HELD-OUT TEST SET EVALUATION")
    logger.info("=" * 80)
    
    # Load best model
    best_model_path = system['trainer'].checkpoint_dir / 'best_model.pt'
    
    evaluator = HeldOutTestEvaluator(
        model_path=str(best_model_path),
        test_loader=system['test_loader'],
        device=system['device'],
        model_class=type(system['model']),
        bootstrap_samples=1000,
        ci_level=0.95,
    )
    
    # Run comprehensive evaluation
    results = evaluator.evaluate()
    
    # Save results
    report_path = Path('test_evaluation_report.json')
    evaluator.save_report(str(report_path))
    logger.info(f"✓ Saved test report to {report_path}")
    
    # Generate visualizations
    evaluator.plot_roc_curve('test_roc_curve.png')
    evaluator.plot_calibration_curve('test_calibration_curve.png')
    evaluator.plot_confusion_matrix('test_confusion_matrix.png')
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SET SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Test AUC-ROC: {results['metrics']['auc']:.6f}")
    logger.info(f"  95% CI: [{results['confidence_intervals']['auc']['lower']:.6f}, "
                f"{results['confidence_intervals']['auc']['upper']:.6f}]")
    
    logger.info(f"\nClinical Performance (optimal threshold):")
    logger.info(f"  Sensitivity: {results['metrics']['sensitivity']:.2%}")
    logger.info(f"  Specificity: {results['metrics']['specificity']:.2%}")
    logger.info(f"  PPV: {results['metrics']['ppv']:.2%}")
    logger.info(f"  NPV: {results['metrics']['npv']:.2%}")
    
    logger.info(f"\nCalibration:")
    logger.info(f"  ECE: {results['calibration']['ece']:.4f}")
    logger.info(f"  Well-calibrated: {results['calibration']['is_well_calibrated']}")
    
    return results


def validate_interpretability(system: dict, results: dict) -> dict:
    """
    Validate that model attention aligns with clinical expectations.
    
    Checks:
    - Center region receives substantial attention (>30%)
    - Attention patterns valid (not diffuse or noisy)
    - Ensemble agreement (consistency across models)
    
    Args:
        system: Output from setup_production_system()
        results: Output from evaluate_on_held_out_test()
        
    Returns:
        Interpretability validation report
    """
    logger.info("=" * 80)
    logger.info("CLINICAL INTERPRETABILITY VALIDATION")
    logger.info("=" * 80)
    
    # Load ensemble
    models = system['models']
    device = system['device']
    
    # Create ensemble explainer
    explainer = EnsembleExplainer(models, device)
    
    # Get a sample batch for visualization
    test_loader = system['test_loader']
    sample_images, sample_labels = next(iter(test_loader))
    
    # Generate explanations
    explanations = explainer.explain(sample_images[:4])  # First 4 samples
    
    logger.info(f"Generated Grad-CAM visualizations for ensemble")
    logger.info(f"  Consensus center-focus: {explanations['mean_center_focus']:.2%}")
    logger.info(f"  Ensemble disagreement: {np.mean(explanations['disagreement']):.4f}")
    
    # Validate attention patterns
    validation = validate_attention_patterns(
        explanations,
        sample_labels.numpy()[:4],
        threshold_center_focus=0.3
    )
    
    logger.info(f"\nAttention Validation:")
    logger.info(f"  Mean center-region focus: {validation['mean_center_focus']:.2%}")
    logger.info(f"  Flagged samples (low focus): {len(validation['flagged_samples'])}")
    logger.info(f"  Validation passed: {validation['validation_passed']}")
    
    # Visualize attention for first sample
    visualizer = AttentionVisualizer()
    attention_viz = visualizer.visualize_attention(
        sample_images[0].cpu().numpy().transpose(1, 2, 0),
        explanations['consensus'][0],
        center_region=True,
    )
    visualizer.save_visualization(attention_viz, Path('attention_visualization.png'))
    logger.info("✓ Saved attention visualization to attention_visualization.png")
    
    return validation


def setup_ood_detection(system: dict) -> CompositeOODDetector:
    """
    Setup out-of-distribution detection system.
    
    Trains detector on validation set to identify domain-shifted samples
    during deployment (e.g., different stain, scanner, tissue type).
    
    Args:
        system: Output from setup_production_system()
        
    Returns:
        Fitted OOD detector ready for deployment
    """
    logger.info("=" * 80)
    logger.info("SETTING UP OOD DETECTION")
    logger.info("=" * 80)
    
    model = system['model']
    device = system['device']
    val_loader = system['val_loader']
    
    # Create composite detector (Mahalanobis + Entropy + Isolation Forest)
    detector = CompositeOODDetector(
        model=model,
        device=device,
        use_mahalanobis=True,
        use_entropy=True,
        use_isolation_forest=True,
    )
    
    # Fit on validation set
    detector.fit(val_loader)
    logger.info("✓ OOD detector fitted on validation set")
    
    # Test on validation set to establish baseline
    val_images, _ = next(iter(val_loader))
    detection_results = detector.detect(val_images)
    
    ood_rate = np.mean(detection_results['is_ood'])
    logger.info(f"\nOOD Detection Baseline (on validation set):")
    logger.info(f"  OOD rate: {ood_rate:.2%} (should be <5% on in-distribution data)")
    logger.info(f"  Mean ensemble score: {np.mean(detection_results['ensemble_score']):.4f}")
    
    # Save detector
    detector.save('ood_detector.pkl')
    logger.info("✓ Saved OOD detector to ood_detector.pkl")
    
    return detector


def generate_regulatory_documentation(system: dict, results: dict) -> None:
    """
    Generate regulatory-compliant documentation.
    
    Creates:
    - Model card (arXiv/NIST format)
    - Design and development report
    - Deployment checklist
    
    Args:
        system: Output from setup_production_system()
        results: Output from evaluate_on_held_out_test()
    """
    logger.info("=" * 80)
    logger.info("GENERATING REGULATORY DOCUMENTATION")
    logger.info("=" * 80)
    
    config = system['config']
    
    # Generate model card
    model_card = {
        'model_name': 'PCam-Ensemble-v1',
        'version': '1.0',
        'created': datetime.now().isoformat(),
        'intended_use': 'Binary classification of histopathology patches (tumor detection)',
        'performance': {
            'test_auc': float(results['metrics']['auc']),
            'test_sensitivity': float(results['metrics']['sensitivity']),
            'test_specificity': float(results['metrics']['specificity']),
        },
        'limitations': [
            'Trained on H&E stained patches from single pathology center',
            'May have domain shift with different stains or scanners',
            'Designed for 96×96 pixel patches from 40× magnification',
        ],
        'hyperparameters': config,
    }
    
    with open('model_card.json', 'w') as f:
        json.dump(model_card, f, indent=2)
    logger.info("✓ Saved model card to model_card.json")
    
    # Note: DEPLOYMENT.md already created for regulatory checklist
    logger.info("✓ Regulatory checklist available in DEPLOYMENT.md")
    logger.info("  Fill out checklist before production deployment")


def demo_deployment_inference(system: dict, detector: CompositeOODDetector) -> None:
    """
    Demonstrate production inference pipeline.
    
    Shows:
    - Loading ensemble for inference
    - Running predictions with uncertainty
    - OOD detection
    - Logging for deployment monitoring
    
    Args:
        system: Output from setup_production_system()
        detector: OOD detector from setup_ood_detection()
    """
    logger.info("=" * 80)
    logger.info("DEMONSTRATION: PRODUCTION INFERENCE")
    logger.info("=" * 80)
    
    # Create ensemble predictor
    ensemble_predictor = EnsemblePredictor(
        model_paths=[str(system['trainer'].checkpoint_dir / 'best_model.pt')],
        device=system['device'],
        use_tta=True,  # Test-time augmentation
    )
    
    # Get sample batch
    test_images, test_labels = next(iter(system['test_loader']))
    
    # Run ensemble inference
    predictions = ensemble_predictor.predict(
        test_images,
        return_uncertainty=True,
    )
    
    logger.info("\nInference Results (first 4 samples):")
    logger.info(f"{'ID':<5} {'True':<6} {'Pred':<6} {'Prob':<8} {'Uncertainty':<12} {'OOD':<6}")
    logger.info("-" * 50)
    
    # Check OOD for samples
    ood_results = detector.detect(test_images[:4])
    
    for i in range(min(4, len(test_images))):
        true_label = int(test_labels[i])
        pred_label = int(predictions['predictions'][i] > 0.5)
        prob = predictions['probabilities'][i]
        uncertainty = predictions['uncertainty'][i]
        is_ood = ood_results['is_ood'][i]
        
        logger.info(
            f"{i:<5} {true_label:<6} {pred_label:<6} "
            f"{prob:<8.4f} {uncertainty:<12.4f} {'Yes' if is_ood else 'No':<6}"
        )
    
    logger.info("\n✓ Inference pipeline demonstration complete")


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 80)
    logger.info("PCam PRODUCTION SYSTEM - COMPLETE WORKFLOW")
    logger.info("=" * 80)
    
    # 1. Setup system
    logger.info("\n[STEP 1] Setting up production system...")
    system = setup_production_system('config.json')
    
    # 2. Train with callbacks
    logger.info("\n[STEP 2] Training with production callbacks...")
    history = train_with_production_callbacks(system)
    
    # 3. Evaluate on test set
    logger.info("\n[STEP 3] Comprehensive test set evaluation...")
    test_results = evaluate_on_held_out_test(system)
    
    # 4. Validate interpretability
    logger.info("\n[STEP 4] Clinical interpretability validation...")
    interp_validation = validate_interpretability(system, test_results)
    
    # 5. Setup OOD detection
    logger.info("\n[STEP 5] Setting up OOD detection...")
    ood_detector = setup_ood_detection(system)
    
    # 6. Generate documentation
    logger.info("\n[STEP 6] Generating regulatory documentation...")
    generate_regulatory_documentation(system, test_results)
    
    # 7. Demo deployment inference
    logger.info("\n[STEP 7] Demonstrating production inference...")
    demo_deployment_inference(system, ood_detector)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ COMPLETE WORKFLOW FINISHED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("1. Review test_evaluation_report.json for final metrics")
    logger.info("2. Verify attention visualizations in attention_visualization.png")
    logger.info("3. Fill out DEPLOYMENT.md regulatory checklist")
    logger.info("4. Prepare 510(k) submission if required")
    logger.info("5. Deploy to production with OOD monitoring")
