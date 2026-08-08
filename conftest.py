"""Make the working tree importable without an install step.

Tests import ``thinair`` from the source tree and ``fakes`` from ``tests/``;
pytest's rootdir insertion covers the first, this covers the second.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tests"))
