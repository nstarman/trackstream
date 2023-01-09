"""Cartesian SOM implementation."""

from __future__ import annotations

# STDLIB
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, final

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u
import numpy as np
from numpy import ndarray, subtract
from numpy.linalg import norm
from scipy.stats import binned_statistic

# LOCAL
from trackstream.track.fit.som.base import SOM1DBase, SOMInfo
from trackstream.track.fit.som.utils import _decay_function, _respace_bins


@final
@dataclass(frozen=True)
class CartesianSOM(SOM1DBase):
    """Initializes a Self-Organizing Map.

    Inspired by the design of [MiniSom]_

    Parameters
    ----------
    nlattice : int
        Number of lattice points (prototypes) in the 1D SOM.

    sigma : float, optional (default=1.0)
        Spread of the neighborhood function, needs to be adequate to the
        dimensions of the map. (at the iteration t we have sigma(t) = sigma / (1
        + t/T) where T is #num_iteration/2)
    learning_rate : initial learning rate
        (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T)
        where T is #num_iteration/2)
    rng : int, optional keyword-only (default=None)
        Random seed to use.

    Notes
    -----
    neighborhood_function : 'gaussian'
        Function that weights the neighborhood of a position in the map.

    activation_distance : 'euclidean'
        Distance used to activate the map.

    References
    ----------
    .. [MiniSom] Giuseppe Vettigli. MiniSom: minimalistic and NumPy-based
        implementation of the Self Organizing Map.
    .. [frankenz] Josh Speagle. Frankenz: a photometric redshift monstrosity.
    """

    info = SOMInfo(
        representation_type=coords.CartesianRepresentation,
        differential_type=coords.CartesianDifferential,
        units=u.StructuredUnit(
            ((u.kpc, u.kpc, u.kpc), (u.km / u.s, u.km / u.s, u.km / u.s)),
            names=(("x", "y", "z"), ("d_x", "d_y", "d_z")),
        ),
    )

    @staticmethod
    def _make_prototypes_from_binned_data(
        data: ndarray, /, nlattice: int, *, byphi: bool = False, maxsep: ndarray | None = None, **_: Any
    ) -> ndarray:
        r"""Initialize prototype vectors from binned data.

        Parameters
        ----------
        data : SkyCoord
        byphi : bool, optional
            Whether to bin by the |Longitude|, or by :math:`\phi=atan(lat/lon)`
        maxsep : Quantity or None, optional keyword-only
            Maximum separation (in data space) between prototypes.
        """
        # TODO? generalize to also kinematics
        # Get coordinate to bin
        x: ndarray = data[:, 0]

        # Determine the binning coordinate
        if byphi:
            x = np.arctan2(data[:, 1], data[:, 0])

        # Create equi-frequency bins
        bins: ndarray = np.interp(x=np.linspace(0, len(x), nlattice + 1), xp=np.arange(len(x)), fp=np.sort(x))

        # Optionally respace the bins to have a maximum separation
        if maxsep is not None:
            # Check respacing is even possible
            if (abs(max(x) - min(x)) / nlattice) > maxsep:
                msg = (
                    f"{nlattice} bins is not enough to cover [{min(x)}, {max(x)}] "
                    f"with a maximum bin separation of {maxsep}"
                )
                raise ValueError(msg)
            # Respace bins
            bins = _respace_bins(  # TODO! speed up
                deepcopy(bins), maxsep, onsky=False, eps=2 * np.finfo(maxsep.dtype).eps
            )

        res = binned_statistic(x, data.T, bins=bins, statistic="median")
        prototypes: ndarray = res.statistic.T

        # When there is no data in a bin, it is set to NaN.
        # This is replaced with the interpolation from nearby points.
        for j, d in enumerate(prototypes.T):
            mask = np.isnan(d)
            prototypes[mask, j] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), d[~mask])

        return prototypes

    # ===============================================================

    def _activation_distance(self, x: ndarray, w: ndarray) -> ndarray:
        distance: ndarray = norm(subtract(x, w), axis=-1)  # works for both q & p
        return distance

    # ---------------------------------------------------------------
    # fitting

    def _update(self, x: ndarray, t: int, max_iteration: int) -> None:
        """Update the locations of the prototypes.

        Parameters
        ----------
        x : (D,) ndarray
            Current point to learn.
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations.
        """
        eta = _decay_function(self.learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = _decay_function(self.sigma, t, max_iteration)
        # improves the performances
        ibmu = self._best_matching_unit_index(x)
        g = self._neighborhood(ibmu, sig) * eta
        self.prototypes[:] += g[:, None] * (x - self.prototypes)
