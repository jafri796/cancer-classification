# PCam Cancer Classification - Regulatory Deployment Checklist

## FDA 510(k) & Clinical ML Submission Requirements

This document serves as a comprehensive checklist for preparing this system for:
- FDA 510(k) pre-market submission (medical device)
- Clinical deployment in a CLIA laboratory
- Peer review and publication
- Regulatory audits and inspections

---

## 1. DATASET & DATA INTEGRITY

### Training Data

- [ ] **Data Provenance**
  - [ ] Source institution(s) documented
  - [ ] Data collection protocol and dates recorded
  - [ ] IRB approval or exemption obtained
  - [ ] Patient consent/waiver documented
  - [ ] Data sharing agreements in place

- [ ] **Dataset Characteristics**
  - [ ] Total number of patches documented: ____
  - [ ] Class distribution (% positive): ____
  - [ ] Number of unique patients: ____
  - [ ] Stain type(s) documented: ____
  - [ ] Histopathology center(s) documented: ____
  - [ ] Scanner model(s) documented: ____

- [ ] **Data Quality**
  - [ ] No duplicate patches within dataset
  - [ ] No patient leakage between splits (cross-checked)
  - [ ] All labels verified by pathologist (blinded review for accuracy)
  - [ ] QC report on label quality (inter-rater agreement, if applicable)
  - [ ] Artefacts/stain issues documented for <5% of patches

### Test Set

- [ ] **Test Set Isolation**
  - [ ] Completely held out during training/development
  - [ ] Never used for hyperparameter tuning
  - [ ] Locked and versioned (git/version control)
  - [ ] Patient-level split (no patient overlap with train/val)

- [ ] **Test Set Characteristics**
  - [ ] Size: ____ samples
  - [ ] Class distribution (% positive): ____
  - [ ] Independent institution (if possible): ____
  - [ ] Different stain/scanner if available: ____

### Data Leakage Prevention

- [ ] **Stain Normalization**
  - [ ] Normalizer fit on training set ONLY
  - [ ] FitSetOrigin tracked and validated
  - [ ] Test set uses pre-fit normalizer (not re-fit)
  - [ ] Audit trail of fit-set origin in logs

- [ ] **Model Calibration**
  - [ ] Temperature scaling fit on validation set
  - [ ] NOT fit on test set (enforced in code)
  - [ ] Rejects "test" set names with error message

- [ ] **Augmentation**
  - [ ] Augmentations applied per-sample (not shared)
  - [ ] Not applied during validation/test

---

## 2. MODEL ARCHITECTURE & DESIGN

### Model Justification

- [ ] **Architecture Choice**
  - [ ] Architectures selected based on published PCam benchmarks
  - [ ] Center-region awareness justified in design docs
  - [ ] Receptive field verified to cover 96×96 input
  - [ ] Design document linked: ____

- [ ] **Center-Region Awareness**
  - [ ] All models implement center 32×32 focus (documented)
  - [ ] For CNNs: spatial attention on center region
  - [ ] For ViTs: positional bias on center patches
  - [ ] Justification: PCam labels based on center 32×32 only

- [ ] **Ensemble Design**
  - [ ] Diversity justified (architecture, initialization, hyperparameters)
  - [ ] Weights for soft voting documented and justified
  - [ ] Weights not trained on test set
  - [ ] Ensemble reduces variance vs single model

### Model Reproducibility

- [ ] **Random Seed**
  - [ ] Seed documented in config: ____
  - [ ] Seed set at: Python, NumPy, PyTorch, CUDA, environment
  - [ ] Reproducibility test: Same seed → identical output (PASS/FAIL)

- [ ] **Model Versioning**
  - [ ] Model weights saved with metadata (epoch, metrics, config)
  - [ ] Config file committed to version control
  - [ ] Model card generated (arXiv template or NIST format)
  - [ ] Git commit hash recorded for training code

---

## 3. TRAINING PROCESS

### Training Configuration

- [ ] **Hyperparameter Justification**
  - [ ] Learning rate justified (typical: 1e-4 to 1e-3)
  - [ ] Batch size justified
  - [ ] Number of epochs documented
  - [ ] Optimizer choice justified
  - [ ] Loss function choice justified

- [ ] **Loss Function**
  - [ ] Loss function: ________
  - [ ] Rationale for choice: ________
  - [ ] Hyperparameters (α, γ for Focal Loss): ________
  - [ ] Ablation study (if applicable): ________

- [ ] **Regularization**
  - [ ] Dropout rates: ________
  - [ ] Weight decay: ________
  - [ ] Early stopping patience: ________
  - [ ] Minimum delta: ________

### Training Integrity

- [ ] **Determinism**
  - [ ] torch.backends.cudnn.deterministic = True
  - [ ] torch.backends.cudnn.benchmark = False
  - [ ] PYTHONHASHSEED set
  - [ ] Reproduction test (same hyperparams → same weights): PASS/FAIL

- [ ] **Monitoring**
  - [ ] Training/validation metrics logged at each epoch
  - [ ] Learning rate logged
  - [ ] Gradient norms logged
  - [ ] Loss curves inspected for pathologies (NaN, divergence, etc.)

- [ ] **Checkpointing**
  - [ ] Best model saved based on validation AUC
  - [ ] Latest checkpoint saved for resumption
  - [ ] Checkpoint metadata includes: epoch, metrics, config
  - [ ] Checkpoints versioned (git-tracked or backup)

- [ ] **Early Stopping**
  - [ ] Enabled: YES/NO
  - [ ] Metric monitored: ________
  - [ ] Patience: ________
  - [ ] Min delta: ________

---

## 4. EVALUATION & METRICS

### Primary Metrics (FDA-aligned)

- [ ] **Area Under ROC Curve (AUC)**
  - [ ] Train AUC: ____
  - [ ] Validation AUC: ____
  - [ ] **Test AUC: ____**
  - [ ] 95% CI (bootstrap): [____, ____]

- [ ] **Clinical Operating Points**
  - [ ] Test Sensitivity @ optimal threshold: ____% (95% CI: [____, ____])
  - [ ] Test Specificity @ optimal threshold: ____% (95% CI: [____, ____])
  - [ ] Threshold value: ____
  - [ ] Rationale for threshold: ________

### Secondary Metrics

- [ ] **Per-Class Metrics**
  - [ ] Precision: ____
  - [ ] Recall: ____
  - [ ] F1-score: ____
  - [ ] Cohen's Kappa: ____

- [ ] **Confusion Matrix (Test Set)**
  - [ ] True Positives: ____
  - [ ] True Negatives: ____
  - [ ] False Positives: ____
  - [ ] False Negatives: ____

- [ ] **Failure Modes**
  - [ ] % False Positives: ____
  - [ ] % False Negatives: ____
  - [ ] FP/FN analysis: ________

### Calibration & Confidence

- [ ] **Model Calibration**
  - [ ] Expected Calibration Error (ECE): ____
  - [ ] Calibration curve plot (saved): YES/NO
  - [ ] Well-calibrated (ECE < 0.1): YES/NO
  - [ ] Temperature scaling applied: YES/NO
  - [ ] Fit set for calibration: ________

- [ ] **Confidence Intervals**
  - [ ] Bootstrap resampling performed (N=1000): YES/NO
  - [ ] 95% CI computed for all metrics: YES/NO
  - [ ] Intervals reported in submission

---

## 5. CLINICAL VALIDATION

### Real-World Performance Scenarios

- [ ] **Different Stains**
  - [ ] If available, tested on non-H&E stain: YES/NO
  - [ ] Performance degradation documented: ____

- [ ] **Different Scanners**
  - [ ] If available, tested on different scanner: YES/NO
  - [ ] Performance degradation documented: ____

- [ ] **Out-of-Distribution Detection**
  - [ ] OOD detector implemented: YES/NO
  - [ ] Detector method: ________
  - [ ] OOD detection rate on test set: ____
  - [ ] False positive rate (OOD on real in-distribution): ____

- [ ] **Robustness Testing**
  - [ ] Adversarial robustness tested: YES/NO
  - [ ] Stain variation handling: ________
  - [ ] Noise robustness: ________

### Clinical Validation Metrics

- [ ] **Sensitivity Requirements**
  - [ ] Target: ≥ 95%
  - [ ] Achieved on test set: ____ % (95% CI: [____, ____])
  - [ ] Met target: YES/NO

- [ ] **Specificity Requirements**
  - [ ] Target: ≥ 90%
  - [ ] Achieved on test set: ____ % (95% CI: [____, ____])
  - [ ] Met target: YES/NO

---

## 6. INTERPRETABILITY & EXPLAINABILITY

### Attention Visualization (Grad-CAM)

- [ ] **Implementation**
  - [ ] Grad-CAM implemented: YES/NO
  - [ ] Target layer: ________
  - [ ] Visualizations generated for test set: YES/NO

- [ ] **Validation of Attention**
  - [ ] Center-region focus verified: YES/NO
  - [ ] Mean center-region attention: ____ % (target: 30-60%)
  - [ ] Attention focused on tumor areas for positive samples: YES/NO
  - [ ] Consensus across ensemble members: YES/NO
  - [ ] Disagreement among ensemble: mean = ____ (low = good)

### Model Card & Documentation

- [ ] **Model Card Created**
  - [ ] Following arXiv/NIST template: YES/NO
  - [ ] Includes: intended use, performance, limitations, bias
  - [ ] Version: ____
  - [ ] Date: ____

- [ ] **Design & Development Documentation**
  - [ ] Architecture justification: YES/NO
  - [ ] Dataset characteristics: YES/NO
  - [ ] Training procedure: YES/NO
  - [ ] Evaluation results: YES/NO
  - [ ] Known limitations: YES/NO

---

## 7. REGULATORY COMPLIANCE

### FDA/CLIA Alignment

- [ ] **Intended Use Statement**
  - [ ] Documented: ________
  - [ ] Patient population defined: ________
  - [ ] Sample type(s): 96×96 histopathology patches from whole slide images
  - [ ] Output: Binary classification (tumor present/absent in center region)

- [ ] **Performance Targets**
  - [ ] Sensitivity ≥ 95%: ACHIEVED/NOT ACHIEVED
  - [ ] Specificity ≥ 90%: ACHIEVED/NOT ACHIEVED
  - [ ] AUC ≥ 0.96: ACHIEVED/NOT ACHIEVED

- [ ] **Risk Analysis**
  - [ ] False positive risk: ________
  - [ ] False negative risk: ________
  - [ ] Mitigation strategies: ________

### Quality Management

- [ ] **Change Control**
  - [ ] Version control (git) implemented: YES/NO
  - [ ] All changes logged with rationale: YES/NO
  - [ ] Model version: ____
  - [ ] Code version/commit: ____

- [ ] **Validation Records**
  - [ ] Training logs preserved: YES/NO
  - [ ] Test results documented: YES/NO
  - [ ] Audit trail of all modifications: YES/NO

- [ ] **Software Configuration**
  - [ ] Dependencies frozen (requirements.txt): YES/NO
  - [ ] Docker image created: YES/NO
  - [ ] Reproducible environment: YES/NO

---

## 8. INFERENCE & DEPLOYMENT

### Inference Speed

- [ ] **Per-Sample Latency**
  - [ ] Single sample: ____ ms (target: <100 ms)
  - [ ] Batch-32: ____ ms (target: <200 ms)
  - [ ] Hardware: GPU _______ or CPU _______

- [ ] **Memory Requirements**
  - [ ] Model size: ____ MB
  - [ ] Peak memory during inference: ____ MB
  - [ ] Suitable for deployment environment: YES/NO

### Deployment Readiness

- [ ] **API Documentation**
  - [ ] Input specification documented: YES/NO
  - [ ] Output specification documented: YES/NO
  - [ ] Error handling documented: YES/NO

- [ ] **Monitoring & Logging**
  - [ ] Predictions logged: YES/NO
  - [ ] OOD scores logged: YES/NO
  - [ ] Performance drift monitoring: YES/NO
  - [ ] Alert system for anomalies: YES/NO

- [ ] **Model Serving**
  - [ ] FastAPI/Flask endpoint: YES/NO
  - [ ] Docker containerization: YES/NO
  - [ ] Kubernetes deployment: YES/NO
  - [ ] Load testing completed: YES/NO

---

## 9. DATA GOVERNANCE & SECURITY

### Privacy & Security

- [ ] **Patient Data Handling**
  - [ ] No PII in training data: YES/NO
  - [ ] Patient IDs anonymized: YES/NO
  - [ ] De-identification audit: YES/NO

- [ ] **Access Control**
  - [ ] Model weights access restricted: YES/NO
  - [ ] Code repository access logged: YES/NO
  - [ ] Deployment credentials managed: YES/NO

- [ ] **Audit Trail**
  - [ ] All model changes logged: YES/NO
  - [ ] Predictions/results logged: YES/NO
  - [ ] Logs retained for ≥3 years: YES/NO

---

## 10. TESTING & VALIDATION

### Unit Tests

- [ ] **Data Pipeline**
  - [ ] Dataset loading: PASS/FAIL
  - [ ] Data augmentation: PASS/FAIL
  - [ ] Stain normalization: PASS/FAIL
  - [ ] No data leakage: PASS/FAIL

- [ ] **Model**
  - [ ] Forward pass: PASS/FAIL
  - [ ] Backward pass (gradients): PASS/FAIL
  - [ ] Inference: PASS/FAIL
  - [ ] Ensemble aggregation: PASS/FAIL

- [ ] **Inference**
  - [ ] Input validation: PASS/FAIL
  - [ ] Output shapes: PASS/FAIL
  - [ ] Edge cases (NaN, Inf): PASS/FAIL
  - [ ] Reproducibility (same seed): PASS/FAIL

### Integration Tests

- [ ] **End-to-End Training**
  - [ ] Full training run completes: PASS/FAIL
  - [ ] Checkpoints saved correctly: PASS/FAIL
  - [ ] Metrics logged: PASS/FAIL

- [ ] **End-to-End Inference**
  - [ ] Load model → infer → output: PASS/FAIL
  - [ ] Batch inference: PASS/FAIL
  - [ ] API endpoint: PASS/FAIL

### Regression Tests

- [ ] **Reproducibility**
  - [ ] Same seed → same model weights: PASS/FAIL
  - [ ] Same seed → same predictions: PASS/FAIL
  - [ ] Determinism enforced: PASS/FAIL

---

## 11. DOCUMENTATION

### Technical Documentation

- [ ] **README.md**
  - [ ] Project overview: YES/NO
  - [ ] Setup instructions: YES/NO
  - [ ] Usage examples: YES/NO
  - [ ] Performance metrics: YES/NO

- [ ] **DEPLOYMENT.md** (this file)
  - [ ] All sections completed: YES/NO

- [ ] **Model Card**
  - [ ] Following arXiv template: YES/NO
  - [ ] All fields populated: YES/NO
  - [ ] Version and date: YES/NO

- [ ] **Design & Development Report**
  - [ ] Architecture justification: YES/NO
  - [ ] Dataset description: YES/NO
  - [ ] Training procedure: YES/NO
  - [ ] Evaluation protocol: YES/NO
  - [ ] Results and performance: YES/NO
  - [ ] Limitations and future work: YES/NO

### Regulatory Documentation

- [ ] **510(k) Submission (if applicable)**
  - [ ] Predicate device identified: YES/NO
  - [ ] Substantial equivalence argument: YES/NO
  - [ ] Performance comparison: YES/NO
  - [ ] Clinical data package: YES/NO

- [ ] **Clinical Evidence**
  - [ ] Publication(s) citing this work: YES/NO
  - [ ] External validation study (if available): YES/NO
  - [ ] Clinical advisory board review: YES/NO

---

## 12. QUALITY ASSURANCE SIGN-OFF

### Approval Sign-Off

| Role | Name | Signature | Date |
|------|------|-----------|------|
| ML Engineer | _____________ | _____________ | _____________ |
| Data Scientist | _____________ | _____________ | _____________ |
| Clinical Advisor | _____________ | _____________ | _____________ |
| QA/Compliance | _____________ | _____________ | _____________ |

### Deployment Approval

| Environment | Status | Approved By | Date |
|-------------|--------|-------------|------|
| Development | [  ] Pass [  ] Fail | _____________ | _____________ |
| Staging | [  ] Pass [  ] Fail | _____________ | _____________ |
| Production | [  ] Pass [  ] Fail | _____________ | _____________ |

---

## APPENDIX: Quick Reference

### Key Metrics (Test Set, Final Model)

```
Primary Metric:
  AUC-ROC: ____ (95% CI: [____, ____])

Clinical Performance (optimal threshold = ____):
  Sensitivity: ____ % (95% CI: [____, ____])
  Specificity: ____ % (95% CI: [____, ____])
  PPV: ____ %
  NPV: ____ %

Secondary Metrics:
  Accuracy: ____ %
  Precision: ____ %
  Recall: ____ %
  F1-score: ____ %
  Cohen's Kappa: ____ %

Model Quality:
  Expected Calibration Error: ____
  Attention center-region focus: ____ %
  OOD detection rate: ____ %

Inference Performance:
  Single sample latency: ____ ms
  Batch-32 latency: ____ ms
  Model size: ____ MB
```

### Contact & Support

- **Technical Issues**: Open GitHub issue or contact ________
- **Regulatory Questions**: Contact ________
- **Clinical Questions**: Contact ________

---

**Last Updated**: ________
**Next Review Date**: ________
