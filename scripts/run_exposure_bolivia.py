"""
Backward-compatible wrapper for Bolivia disaster exposure pipeline.
Redirects to scripts/run_exposure.py --event bolivia.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_exposure import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--event", "bolivia"]
    main()
