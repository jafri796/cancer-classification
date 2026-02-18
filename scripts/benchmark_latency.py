#!/usr/bin/env python
"""
Benchmark inference latency for PCam models.

Measures p50/p95/p99 latency across many synthetic samples to verify
the <200ms per-image target on GPU (or CPU).

Usage:
    python scripts/benchmark_latency.py --model resnet50_cbam --device cuda
    python scripts/benchmark_latency.py --model deit_small --device cpu --n-samples 200
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    create_center_aware_resnet50,
    create_resnet50_cbam,
    create_efficientnet,
    create_vit,
    create_deit_small,
)
from src.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_FACTORIES = {
    "resnet50_se": create_center_aware_resnet50,
    "resnet50_cbam": create_resnet50_cbam,
    "efficientnet_b3": lambda cfg: create_efficientnet({**cfg, "architecture": "efficientnet-b3"}),
    "vit_b16": create_vit,
    "deit_small": create_deit_small,
}


def benchmark(model: torch.nn.Module, device: torch.device, n_samples: int, warmup: int = 20):
    """Run inference benchmark and return per-sample latencies in ms."""
    model.eval()
    dummy = torch.randn(1, 3, 96, 96, device=device)

    # Warmup passes (exclude from measurement)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(n_samples):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

    return np.array(latencies)


def main():
    parser = argparse.ArgumentParser(description="Benchmark PCam model inference latency")
    parser.add_argument("--model", type=str, default="resnet50_cbam", choices=list(MODEL_FACTORIES.keys()))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=500, help="Number of inference passes")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup passes (excluded)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    logger.info(f"Benchmarking {args.model} on {device} ({args.n_samples} samples, {args.warmup} warmup)")

    cfg = {"pretrained": False, "num_classes": 1}
    model = MODEL_FACTORIES[args.model](cfg).to(device)

    latencies = benchmark(model, device, args.n_samples, args.warmup)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mean = latencies.mean()
    std = latencies.std()

    target = 200.0  # ms

    print()
    print("=" * 60)
    print(f"LATENCY BENCHMARK: {args.model} on {device}")
    print("=" * 60)
    print(f"  Samples:  {args.n_samples}")
    print(f"  Mean:     {mean:.2f} ms  (std {std:.2f})")
    print(f"  p50:      {p50:.2f} ms")
    print(f"  p95:      {p95:.2f} ms")
    print(f"  p99:      {p99:.2f} ms")
    print(f"  Min:      {latencies.min():.2f} ms")
    print(f"  Max:      {latencies.max():.2f} ms")
    print(f"  Target:   <{target:.0f} ms")
    status = "PASS" if p95 < target else "FAIL"
    print(f"  Status:   {status} (p95 {'<' if p95 < target else '>='} {target:.0f} ms)")
    print("=" * 60)

    if status == "FAIL":
        logger.warning(f"p95 latency {p95:.2f}ms exceeds {target:.0f}ms target")
        sys.exit(1)


if __name__ == "__main__":
    main()
