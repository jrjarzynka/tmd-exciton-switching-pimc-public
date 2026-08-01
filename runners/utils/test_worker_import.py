#!/usr/bin/env python3
import sys
import os
import traceback

# ensure local package is importable for workers
# Search a few parent levels for numerics/tmd_pimc (robust to runners/<sub>/ nesting).
_here_dir = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = None
_walk = _here_dir
for _ in range(5):
    _candidate = os.path.join(_walk, "numerics")
    if os.path.isdir(os.path.join(_candidate, "tmd_pimc")):
        _CODE_DIR = _candidate
        break
    _legacy = os.path.join(_walk, "code")
    if os.path.isdir(os.path.join(_legacy, "tmd_pimc")):
        _CODE_DIR = _legacy
        break
    _walk = os.path.dirname(_walk)
if _CODE_DIR is None:
    raise ImportError(
        f"Could not locate numerics/tmd_pimc within 5 parent levels of {_here_dir}."
    )
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from concurrent.futures import ProcessPoolExecutor

def check():
    try:
        import tmd_pimc
        return ("OK", getattr(tmd_pimc, "__file__", "<package>"))
    except Exception:
        return ("ERR", traceback.format_exc())

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2) as ex:
        futures = [ex.submit(check) for _ in range(2)]
        for f in futures:
            print(f.result())

