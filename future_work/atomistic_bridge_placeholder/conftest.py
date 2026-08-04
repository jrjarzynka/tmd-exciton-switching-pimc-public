# conftest.py (future_work/atomistic_bridge_placeholder/conftest.py)
#
# Mirrors the top-level repo's conftest.py convention: adds this
# directory's own package root to sys.path so `import moire_pipeline`
# resolves regardless of the current working directory pytest is
# invoked from, without needing PYTHONPATH set by hand.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
