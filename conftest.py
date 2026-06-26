"""Make the repo-root `embers` and `bench` packages importable in tests
without an install."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
