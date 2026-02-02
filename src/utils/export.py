"""
Backwards-compatible export helpers.

Historically some code imported ``src.utils.export.export_model`` even
though the primary implementation lives in ``src.inference.export``.
To avoid breaking those imports, this module provides a thin wrapper
around the canonical exporter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch.nn as nn

from src.inference.export import ModelExporter as _ModelExporter


def export_model(
    model: nn.Module,
    example_input,
    output_dir: str | Path,
    model_name: str = "model",
    formats: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Export ``model`` using the canonical inference exporter.

    Args:
        model: Trained PyTorch model to export.
        example_input: Representative input tensor for tracing/verification.
        output_dir: Directory where exported artifacts will be written.
        model_name: Logical name used for output file prefixes.
        formats: Optional list of formats; if ``None`` all supported formats
                 are produced. This parameter is accepted for compatibility
                 with older callers but is currently ignored – the underlying
                 exporter decides which formats to emit.

    Returns:
        A dictionary of export reports keyed by format name.
    """
    exporter = _ModelExporter(
        model=model,
        example_input=example_input,
        output_dir=output_dir,
        model_name=model_name,
    )

    if not formats or "all" in formats:
        return exporter.export_all()

    reports: Dict[str, Dict] = {}
    if "torchscript" in formats:
        reports["torchscript"] = exporter.export_torchscript()
    if "onnx" in formats:
        reports["onnx"] = exporter.export_onnx()
    if "quantized" in formats:
        reports["quantized"] = exporter.export_quantized()

    return reports

