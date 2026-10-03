import sys
from pathlib import Path

# Ensure project root is always on sys.path during test collection and execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
