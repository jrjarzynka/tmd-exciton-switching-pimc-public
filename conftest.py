"""Make the ``tmd_pimc`` package importable from the repository root.

Allows the test suite and the runner scripts to be executed without an
editable install; ``pip install -e .`` remains the recommended route.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "numerics"))
