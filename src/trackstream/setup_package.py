# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Set up package."""

from __future__ import annotations

import logging
import sys

__all__: list[str] = []


PY_GE_310 = sys.version_info >= (3, 10)

try:
    import tqdm  # noqa: F401
except ImportError:
    HAS_TQDM = False
else:
    HAS_TQDM = True


# Logging
logger = logging.getLogger("trackstream")
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
