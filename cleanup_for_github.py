#!/usr/bin/env python
"""
Cleanup script for GitHub release preparation.
Run from project root: python cleanup_for_github.py
"""
import shutil
from pathlib import Path

ROOT = Path(".")

print("=" * 50)
print("PCam Project GitHub Cleanup")
print("=" * 50)

# Phase 1: Delete deprecated files
print("\n[Phase 1] Deleting deprecated files...")
deprecated = [
    "src/inference/model_export.py",
    "src/validation/clinical_validation_new.py",
]
for f in deprecated:
    p = ROOT / f
    if p.exists():
        p.unlink()
        print(f"  ✓ Deleted: {f}")
    else:
        print(f"  - Already gone: {f}")

# Phase 2: Move workflow scripts to examples/
print("\n[Phase 2] Moving workflow scripts to examples/...")
examples_dir = ROOT / "examples"
examples_dir.mkdir(exist_ok=True)

workflows = [
    "workflow_1_api_test.py",
    "workflow_2_4_models_ensemble.py",
    "workflow_5_mlflow.py",
    "workflow_6_mock_data.py",
    "workflow_7_tests.py",
    "workflow_8_9_validation.py",
    "workflow_10_analysis.py",
    "workflow_production_demo.py",
    "run_all_workflows.py",
    "validate_system.py",
]
for f in workflows:
    src = ROOT / f
    if src.exists():
        shutil.move(str(src), str(examples_dir / f))
        print(f"  ✓ Moved: {f} -> examples/")
    else:
        print(f"  - Not found: {f}")

# Phase 3: Move docs to docs/
print("\n[Phase 3] Moving documentation to docs/...")
docs = ["QUICKSTART.md", "IMPLEMENTATION_SUMMARY.md", "DEPLOYMENT.md"]
for f in docs:
    src = ROOT / f
    dst = ROOT / "docs" / f
    if src.exists():
        shutil.move(str(src), str(dst))
        print(f"  ✓ Moved: {f} -> docs/")
    else:
        print(f"  - Not found: {f}")

print("\n" + "=" * 50)
print("✓ Cleanup complete!")
print("=" * 50)
print("\nNext steps:")
print("  git add -A")
print('  git commit -m "refactor: prepare for GitHub release"')
print("  git push origin main")
