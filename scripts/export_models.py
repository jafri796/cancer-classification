#!/usr/bin/env python
"""
Export trained models to production formats.

Supports exporting to:
- ONNX (cross-platform inference)
- TorchScript (C++ deployment)
- Quantized INT8 (reduced size and faster CPU inference)

Usage:
    python scripts/export_models.py --checkpoint experiments/best_model.pt --formats onnx torchscript
    python scripts/export_models.py --checkpoint experiments/best_model.pt --formats quantized --output-dir models
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.export import ModelExporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ["onnx", "torchscript", "quantized", "all"]


def load_model_from_checkpoint(checkpoint_path: Path) -> torch.nn.Module:
    """Load model from a training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_class = checkpoint.get("model_class", None)
        model_config = checkpoint.get("model_config", {})
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model_class = checkpoint.get("model_class", None)
        model_config = checkpoint.get("config", {})
    else:
        # Assume it's just the state dict
        state_dict = checkpoint
        model_class = None
        model_config = {}
    
    # Try to reconstruct model
    if model_class:
        from src.models import (
            create_center_aware_resnet50,
            create_resnet50_cbam,
            create_efficientnet,
            create_vit,
        )
        
        factories = {
            "CenterAwareResNet50SE": create_center_aware_resnet50,
            "ResNet50CBAM": create_resnet50_cbam,
            "CenterAwareEfficientNet": create_efficientnet,
            "CenterAwareViT": create_vit,
        }
        
        if model_class in factories:
            model = factories[model_class](**model_config)
            model.load_state_dict(state_dict)
            return model
    
    # If we can't determine the model type, raise an error
    raise ValueError(
        "Cannot determine model architecture from checkpoint. "
        "Please ensure the checkpoint contains 'model_class' metadata."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export trained models to production formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export to ONNX
    python scripts/export_models.py --checkpoint best_model.pt --formats onnx

    # Export to multiple formats
    python scripts/export_models.py --checkpoint best_model.pt --formats onnx torchscript quantized

    # Export all formats to specific directory
    python scripts/export_models.py --checkpoint best_model.pt --formats all --output-dir models/production
        """
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=SUPPORTED_FORMATS,
        default=["onnx"],
        help="Export formats (default: onnx)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Output directory for exported models (default: models)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name prefix for exported files (default: checkpoint filename)"
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported models produce same outputs as original"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint
    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model name
    model_name = args.model_name or args.checkpoint.stem
    
    # Expand 'all' format
    if "all" in args.formats:
        formats = ["onnx", "torchscript", "quantized"]
    else:
        formats = args.formats
    
    logger.info(f"Loading model from: {args.checkpoint}")
    try:
        model = load_model_from_checkpoint(args.checkpoint)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    logger.info(f"Exporting to formats: {formats}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create exporter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exporter = ModelExporter(model, device=device)
    
    # Example input for tracing
    example_input = torch.randn(1, 3, 96, 96).to(device)
    
    results = {}
    
    for fmt in formats:
        try:
            logger.info(f"\nExporting to {fmt}...")
            
            if fmt == "onnx":
                output_path = args.output_dir / f"{model_name}.onnx"
                report = exporter.export_onnx(
                    output_path,
                    example_input,
                    opset_version=args.opset_version,
                )
                results["onnx"] = report
                
            elif fmt == "torchscript":
                output_path = args.output_dir / f"{model_name}.jit"
                report = exporter.export_torchscript(output_path, example_input)
                results["torchscript"] = report
                
            elif fmt == "quantized":
                output_path = args.output_dir / f"{model_name}_int8.pt"
                report = exporter.export_quantized(output_path)
                results["quantized"] = report
            
            logger.info(f"  ✓ {fmt} export complete")
            
        except Exception as e:
            logger.error(f"  ✗ {fmt} export failed: {e}")
            results[fmt] = {"error": str(e)}
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("EXPORT SUMMARY")
    logger.info("=" * 50)
    
    for fmt, report in results.items():
        if "error" in report:
            logger.info(f"  {fmt}: FAILED - {report['error']}")
        else:
            size_mb = report.get("size_mb", "N/A")
            logger.info(f"  {fmt}: SUCCESS ({size_mb} MB)")
    
    # Verification
    if args.verify and "error" not in results.get("onnx", {"error": True}):
        logger.info("\nVerifying ONNX model outputs...")
        try:
            import onnxruntime as ort
            
            onnx_path = args.output_dir / f"{model_name}.onnx"
            session = ort.InferenceSession(str(onnx_path))
            
            # Compare outputs
            with torch.no_grad():
                torch_output = model(example_input.cpu()).numpy()
            
            onnx_input = {session.get_inputs()[0].name: example_input.cpu().numpy()}
            onnx_output = session.run(None, onnx_input)[0]
            
            max_diff = abs(torch_output - onnx_output).max()
            logger.info(f"  Max output difference: {max_diff:.6f}")
            
            if max_diff < 1e-4:
                logger.info("  ✓ Verification PASSED")
            else:
                logger.warning("  ⚠ Verification WARNING: outputs differ more than expected")
                
        except ImportError:
            logger.warning("  onnxruntime not installed, skipping verification")
        except Exception as e:
            logger.error(f"  Verification failed: {e}")
    
    logger.info(f"\n✓ Export complete. Files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
