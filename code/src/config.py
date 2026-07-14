"""Shared paths and configuration for the grading benchmark."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Inputs
CSV_PATH = ROOT / "other.csv"
IMAGES_DIR = ROOT / "other"

# Working data
DATA_DIR = ROOT / "data"
ERASED_DIR = DATA_DIR / "erased"          # red-erased images fed to the grader
OVERLAY_DIR = DATA_DIR / "redmask_overlay"  # red-mask overlays for CV validation
CACHE_DIR = DATA_DIR / "cache"            # raw grading responses keyed by id

# Outputs
OUTPUTS_DIR = ROOT / "outputs"
BY_SUBJECT_DIR = OUTPUTS_DIR / "by_subject"
GROUND_TRUTH_CSV = OUTPUTS_DIR / "ground_truth.csv"
METRICS_CSV = OUTPUTS_DIR / "metrics.csv"
DISAGREEMENTS_CSV = OUTPUTS_DIR / "disagreements.csv"
GRADES_CSV = OUTPUTS_DIR / "grades.csv"

for _d in (ERASED_DIR, OVERLAY_DIR, CACHE_DIR, OUTPUTS_DIR, BY_SUBJECT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Model
MODEL = "gemini-2.5-flash"

# Retry / backoff for rate limits (429) and transient server errors (503).
# Waits grow 8, 16, 32, 60, 60, 60 s — enough to ride out a per-minute quota window.
MAX_RETRIES = 7
BACKOFF_BASE = 8.0
BACKOFF_CAP = 60.0


def get_api_key() -> str:
    """Read the Vertex AI API key from the environment or a .env file."""
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "API_KEY", "VERTEX_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val.strip()
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "API_KEY", "VERTEX_API_KEY"):
                return v.strip().strip('"').strip("'")
    raise RuntimeError(
        "No API key found. Set GEMINI_API_KEY (or API_KEY) in the environment "
        "or in a .env file at the project root."
    )
