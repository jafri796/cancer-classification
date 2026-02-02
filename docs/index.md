# PCam Classification Documentation

FDA-compliant center-aware deep learning for histopathology tumor detection.

## Documentation Index

- [Quick Start Guide](QUICKSTART.md) - Get up and running in 5 minutes
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Technical architecture overview
- [Deployment Guide](DEPLOYMENT.md) - Production deployment and FDA compliance
- [API Reference](API.md) - REST API documentation
- [Model Architecture](MODELS.md) - Center-aware model design

## Quick Links

### Installation
```bash
pip install -e ".[all]"
```

### Training
```bash
python scripts/train.py --config config/training_config.yaml
```

### Evaluation
```bash
python scripts/evaluate_model.py --checkpoint experiments/best_model.pt
```

### API Server
```bash
uvicorn deployment.api.app:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
cancer-classification/
├── config/          # YAML configuration files
├── deployment/      # API and Kubernetes manifests
├── docs/            # Documentation
├── scripts/         # CLI tools
├── src/             # Source code
│   ├── configs/     # Pydantic schemas
│   ├── data/        # Dataset and preprocessing
│   ├── inference/   # Production inference
│   ├── models/      # Model architectures
│   ├── training/    # Training pipeline
│   ├── validation/  # Evaluation and clinical validation
│   └── utils/       # Utilities
└── tests/           # Test suite
```

## License

MIT License - See [LICENSE](../LICENSE) for details.
