# conftest.py  (katalog główny projektu, np.
# ".../v1.8b_2D_RK_PIMC_validation/conftest.py")
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "numerics"))
