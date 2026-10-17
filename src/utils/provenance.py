"""
Provenance stamp for evaluation outputs: which code, configuration, rubric, answer key and penalty profile
produced a set of marks, so that two evaluations can be compared or a result reproduced.
"""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

# Code whose behaviour decides the marks
SCORING_CODE = (
    "src/pipeline/answer_segmenter.py",
    "src/pipeline/stage4_modes.py",
    "src/pipeline/stage4_evaluator.py",
    "src/pipeline/penalty_calculator.py",
)


def file_sha256(path: Optional[str]) -> Optional[str]:
    """First 16 hex digits of the file's SHA-256, or None if the file is missing."""
    if not path or not Path(path).is_file():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def git_commit() -> Optional[str]:
    """Current commit, with '+dirty' when tracked files have uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=5).stdout.strip()
        return (sha + ("+dirty" if dirty else "")) or None
    except Exception:
        return None


def build_provenance(config: Any, rubric_path: Optional[str], answer_key_path: Optional[str],
                     penalty_path: Optional[str]) -> Dict[str, Any]:
    try:
        cfg_dump = config.model_dump() if hasattr(config, "model_dump") else dict(config or {})
        cfg_hash = hashlib.sha256(json.dumps(cfg_dump, sort_keys=True, default=str).encode()).hexdigest()[:16]
    except Exception:
        cfg_hash = None
    return {
        "git_commit": git_commit(),
        "config_sha256": cfg_hash,
        "rubric": {"path": rubric_path, "sha256": file_sha256(rubric_path)},
        "answer_key": {"path": answer_key_path, "sha256": file_sha256(answer_key_path)},
        "penalty_profile": {"path": penalty_path, "sha256": file_sha256(penalty_path)},
        "scoring_code_sha256": {p: file_sha256(p) for p in SCORING_CODE},
    }
