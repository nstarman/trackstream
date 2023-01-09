# see LICENSE.rst

"""Classes and functions for working with stellar streams."""

# STDLIB
from importlib.metadata import version as _get_version

# LOCAL
from trackstream.stream.core import StreamArm
from trackstream.stream.stream import Stream

__all__ = ["StreamArm", "Stream"]
__version__ = _get_version(__name__)


# ===================================================================

# Fill in attrs, etc.
# isort: split
# LOCAL
from trackstream import frame, setup_package  # noqa: F401
from trackstream.io.register import UnifiedIOEntryPointRegistrar

UnifiedIOEntryPointRegistrar(data_class=StreamArm, group="trackstream.io.StreamArm.from_format", which="reader").run()
# clean up
del UnifiedIOEntryPointRegistrar

del _get_version
