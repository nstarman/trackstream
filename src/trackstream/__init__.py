# see LICENSE.rst

"""This module contains class and functions for working with stellar streams."""

# LOCAL
from trackstream import utils  # noqa: F401
from trackstream.stream.core import StreamArm
from trackstream.stream.stream import Stream

__all__ = ["StreamArm", "Stream"]


# ===================================================================

# Fill in attrs, etc.
# isort: split
# LOCAL
from trackstream import frame  # noqa: F401
from trackstream import setup_package  # noqa: F401

# ===================================================================
# Register I/O

# isort: split
# LOCAL
from trackstream.io.register import UnifiedIOEntryPointRegistrar

UnifiedIOEntryPointRegistrar(data_class=StreamArm, group="trackstream.io.StreamArm.from_format", which="reader").run()
# clean up
del UnifiedIOEntryPointRegistrar
