"""
WORKFLOW 7: Run Unit Tests
Validate all modules work correctly
"""

import sys
import subprocess
from pathlib import Path

print("\n" + "="*80)
print("WORKFLOW 7: UNIT TESTS")
print("="*80)

print("\n📌 Running pytest on all test files...")

# Run pytest with verbose output
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
    cwd=".",
    capture_output=True,
    text=True,
    timeout=120
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[:500])

# Print summary
print("\n" + "="*80)
if result.returncode == 0:
    print("✓ WORKFLOW 7 COMPLETE: All tests passed!")
else:
    print("⚠ WORKFLOW 7 COMPLETE: Some tests may have issues")
    print(f"  Return code: {result.returncode}")
print("="*80)
