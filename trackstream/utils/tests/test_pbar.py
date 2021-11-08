# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.utils.misc`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import pytest

# LOCAL
from trackstream.setup_package import HAS_TQDM
from trackstream.utils.pbar import _NoOpPBar, get_progress_bar

if HAS_TQDM:
    # THIRD PARTY
    import tqdm

##############################################################################
# TESTS
##############################################################################


@pytest.mark.skipif(HAS_TQDM, reason="testing when tqdm is not installed.")
def test_get_progress_bar_tqdm_not_installed(caplog):
    """Test ``get_progress_bar`` when TQDM is NOT installed."""
    pbar = get_progress_bar(display=True, total=100)
    assert isinstance(pbar, _NoOpPBar)
    assert "tqdm library" in caplog.text


@pytest.mark.skipif(not HAS_TQDM, reason="requires tqdm.")
def test_get_progress_bar_():
    """Test ``get_progress_bar`` when TQDM is installed."""
    pbar = get_progress_bar(display=False, total=100)
    assert isinstance(pbar, _NoOpPBar)

    pbar = get_progress_bar(display=True, total=100)
    assert isinstance(pbar, tqdm.tqdm)

    # pbar = get_progress_bar(display="gui", total=100)
    # assert isinstance(pbar, tqdm.tqdm_gui)


##############################################################################
# END
