# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.preprocess.som`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import numpy as np
import pytest

# LOCAL
from trackstream.preprocess.som import SelfOrganizingMap1D, asymptotic_decay

##############################################################################
# TESTS
##############################################################################


class TestSelfOrganizingMap1D:
    """Test `trackstream.preprocess.som.SelfOrganizingMap1D`."""

    def setup_class(self):
        """Setup class for testing."""
        self.som_cls = SelfOrganizingMap1D
        self.nlattice = 10
        self.nfeature = 3
        self.sigma = 0.1
        self.learning_rate = 0.3
        self.decay_function = "asymptotic"

    @pytest.fixture
    def som(self):
        return self.som_cls(
            self.nlattice,
            self.nfeature,
            sigma=self.sigma,
            learning_rate=self.learning_rate,
        )

    # ===============================================================
    # Method Tests

    def test_init(self):
        # minimum mandatory arguments
        nlattice, nfeature = 10, 3
        sigma = 0.3
        som = self.som_cls(nlattice, nfeature, sigma=sigma)
        assert som._nlattice == nlattice
        assert som._nfeature == nfeature
        assert som._sigma == 0.3

        # different arguments

    # ---------------------------------------------------------------
    # properties

    def test_nlattice(self, som):
        """Test :attr:`trackstream.preprocess.som.SelfOrganizingMap1D.nlattice`."""
        assert som.nlattice is som._nlattice
        assert som.nlattice == self.nlattice

    def test_nfeature(self, som):
        """Test :attr:`trackstream.preprocess.som.SelfOrganizingMap1D.nfeature`."""
        assert som.nfeature is som._nfeature
        assert som.nfeature == self.nfeature

    def test_sigma(self, som):
        """Test :attr:`trackstream.preprocess.som.SelfOrganizingMap1D.sigma`."""
        assert som.sigma is som._sigma
        assert som.sigma == self.sigma

    def test_learning_rate(self, som):
        """Test :attr:`trackstream.preprocess.som.SelfOrganizingMap1D.learning_rate`."""
        assert som.learning_rate is som._learning_rate
        assert som.learning_rate == self.learning_rate

    def test_decay_function(self, som):
        """Test :attr:`trackstream.preprocess.som.SelfOrganizingMap1D.decay_function`."""
        assert som.decay_function is som._decay_function
        assert som.decay_function is asymptotic_decay

    # ---------------------------------------------------------------
    # methods

    # "._activate()" is not meant to be called

    def test_activation_distance(self, som):
        assert som._activation_distance([1, 2], [1, 2]) == 0
        assert som._activation_distance([1, 2], [2, 3]) == np.sqrt(2)

    def test_distance_from_weights(self, som):
        """Test ``SelfOrganizingMap1D._distance_from_weights``."""
        # test when everything aligns
        data = np.ones((self.nlattice, self.nfeature)) * 10
        data[:, 0] -= 1
        data[:, 2] += 1

        distance = som._distance_from_weights(data)
        assert distance.shape == (self.nlattice, self.nlattice)

        # and errors when not
        with pytest.raises(ValueError, match="shapes"):
            som._distance_from_weights(data[::2, ::2])

    @pytest.mark.skip("TODO!")
    def test_pca_weights_init(self, som):
        """Test ``SelfOrganizingMap1D.pca_weights_init``."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_binned_weights_init(self, som):
        """Test ``SelfOrganizingMap1D.binned_weights_init``."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_quantization(self, som):
        """Test ``SelfOrganizingMap1D.quantization``."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_quantization_error(self, som):
        """Test ``SelfOrganizingMap1D.quantization_error``."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_train(self, som):
        """Test :meth:`trackstream.preprocess.som.SelfOrganizingMap1D.train`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_neighborhood(self, som):
        """Test :meth:`trackstream.preprocess.som.SelfOrganizingMap1D.neighborhood`."""
        som.neighborhood(10, 0.3)
        assert False

    @pytest.mark.skip("TODO!")
    def test_update(self, som):
        """Test :meth:`trackstream.preprocess.som.SelfOrganizingMap1D.update`."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_winner(self, som):
        """Test :meth:`trackstream.preprocess.som.SelfOrganizingMap1D.winner`."""
        assert False


# /class

#####################################################################


@pytest.mark.parametrize("learning_rate", [0, 1, 2])
@pytest.mark.parametrize("iteration", [0, 1, 2])
@pytest.mark.parametrize("max_iter", [10, 10, 10])
def test_asymptotic_decay(learning_rate, iteration, max_iter):
    """Test :func:`trackstream.preprocess.som.asymptotic_decay`."""
    expected = learning_rate / (1.0 + (2.0 * iteration / max_iter))
    got = asymptotic_decay(learning_rate, iteration, max_iter)
    assert got == expected


# /def


@pytest.mark.skip("TODO!")
def test_reorder_visits():
    """Test :func:`trackstream.preprocess.som.reorder_visits`."""
    assert False


# /def


# @pytest.mark.skip("TODO!")
# def test_prepare_SOM():
#     """Test :func:`trackstream.preprocess.som.prepare_SOM`."""
#     assert False
#
#
# # /def


@pytest.mark.skip("TODO!")
def test_order_data():
    """Test :func:`trackstream.preprocess.som.order_data`."""
    assert False


# /def

##############################################################################
# END
