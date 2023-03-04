"""Testing :mod:`~trackstream.utils.pbar`."""


import pytest

from trackstream.setup_package import HAS_TQDM
from trackstream.utils.pbar import _NoOpPBar, get_progress_bar

if HAS_TQDM:
    import tqdm

##############################################################################
# TESTS
##############################################################################


class Test_NoOpPBar:
    """Test :func:`trackstream.utils.pbar._NoOpPBar`."""

    def test_init(self):
        pbar = _NoOpPBar()
        assert isinstance(pbar, _NoOpPBar)

    def test_running(self):
        with _NoOpPBar() as pbar:
            for _i in range(10):
                pbar.update(1)

        assert isinstance(pbar, _NoOpPBar)


@pytest.mark.skipif(HAS_TQDM, reason="testing when tqdm is not installed.")
def test_get_progress_bar_tqdm_not_installed(caplog):
    """Test ``get_progress_bar`` when :mod:`tqdm` is NOT installed."""
    pbar = get_progress_bar(display=True, total=100)
    assert isinstance(pbar, _NoOpPBar)
    assert "tqdm library" in caplog.text


@pytest.mark.skipif(not HAS_TQDM, reason="requires tqdm.")
def test_get_progress_bar_():
    """Test :func:`trackstream.utils.pbar.get_progress_bar`.

    Test :func:`trackstream.utils.pbar.get_progress_bar` when :mod:`tqdm` is
    installed. :mod:`tqdm` itself is well tested, so no further tests are needed here.
    """
    pbar = get_progress_bar(display=False, total=100)
    assert isinstance(pbar, _NoOpPBar)

    pbar = get_progress_bar(display=True, total=100)
    assert isinstance(pbar, tqdm.tqdm)
