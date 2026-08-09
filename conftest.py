"""Make the working tree importable without an install step.

Tests import ``thinair`` from the source tree and ``fakes`` from ``tests/``;
pytest's rootdir insertion covers the first, this covers the second.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "tests"))

# The suite is hermetic: no test writes to a shared on-disk store.  Tests of
# the store itself construct SqliteLedger explicitly against tmp_path.
os.environ.setdefault("THINAIR_STORE", "off")
