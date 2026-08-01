# moire_pipeline/utils.py
from pathlib import Path
import json
import hashlib
import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def save_npz_with_meta(out_path: str | Path, payload: Dict[str, Any], meta: Dict[str, Any]):
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outp, **payload)
    meta_path = outp.with_suffix(".json")
    meta = dict(meta)
    meta.setdefault("generated_by", __name__)
    meta.setdefault("format", "npz_compressed")
    meta_path.write_text(json.dumps(meta, indent=2))
    sha = hashlib.sha256(outp.read_bytes()).hexdigest()
    outp.with_suffix(".npz.sha256").write_text(sha)
    logger.info("Saved %s (sha256: %s) and metadata %s", outp, sha, meta_path)
    return outp

