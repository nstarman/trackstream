"""SOM on the sphere."""

from __future__ import annotations

# STDLIB
from copy import deepcopy
from dataclasses import dataclass
from math import pi
from typing import Any, final

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u
import numpy as np
from astropy.coordinates.angle_utilities import angular_separation
from erfa import s2pv
from numpy import ndarray, subtract
from numpy.linalg import norm
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import distance_matrix
from scipy.stats import binned_statistic

# LOCAL
from trackstream.track.fit.som.base import SOM1DBase, SOMInfo
from trackstream.track.fit.som.utils import _decay_function, _respace_bins, wrap_at
from trackstream.track.fit.utils import offset_by, position_angle

twopi = 2 * pi
halfpi = pi / 2


@final
@dataclass(frozen=True)
class USphereSOM(SOM1DBase):
    """Initializes a Self-Organizing Map.

    Inspired by the design of [MiniSom]_

    Parameters
    ----------
    prototypes : (N, F) ndarray
        The N prototype vectors of F features.
        F can be:

        - 2: (longitude, latitude)
        - 4: (longitude, latitude, d_longitude, d_latitude)

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
        representation_type=coords.UnitSphericalRepresentation,
        differential_type=coords.UnitSphericalCosLatDifferential,
        units=u.StructuredUnit(
            ((u.rad, u.rad), (u.mas / u.yr, u.mas / u.yr)),
            names=(("lon", "lat"), ("d_lon_coslat", "d_lat")),
        ),
    )

    @staticmethod
    def _make_prototypes_from_binned_data(
        data: ndarray,
        /,
        nlattice: int,
        *,
        byphi: bool = False,
        maxsep: ndarray | None = None,
        **_: Any,
    ) -> ndarray:
        r"""Initialize prototype vectors from binned data.

        Parameters
        ----------
        data : (N, D) ndarray
        byphi : bool, optional
            Whether to bin by the |Longitude|, or by :math:`\phi=atan(lat/lon)`
        maxsep : ndarray or None, optional keyword-only
            Maximum separation (in data space) between prototypes.
        """
        # Get coordinate to bin
        # This is most easily done as a NON-structured array
        x: ndarray = data[:, 0]

        # Determine the binning coordinate
        if byphi:
            x = np.arctan2(data[:, 1], data[:, 0])  # radians
        else:
            x = wrap_at(x, pi)

            # Unwrap, as best we can
            # Start by separating the populations so we can grab all wrapped points
            d = distance_matrix(x[:, None], x[:, None])
            d[d == 0] = np.nan
            t = np.mean(np.nanmean(d, axis=0))  # typical separation, upweighted by sep groups
            label = fclusterdata(x[:, None], t=t, criterion="distance") - 1
            # TODO! this is only for 2 pops, what if 3+?
            x0, x1, lesser = x[label == 0], x[label == 1], 0
            # determine if there's more than one group. There might be only 1.
            groups = bool(len(x0) >= 1 and len(x1) >= 1)
            if groups and min(x1) < min(x0):  # rearrange to correct order
                lesser = 1
                x0, x1 = x1, x0
            if groups and angular_separation(min(x0), 0, max(x1), 0) < t:
                idx = label == lesser

                x[idx] = x[idx] + 2 * pi
                data[idx, 0] = data[idx, 0] + 2 * pi

        # Create equi-frequency bins
        # https://www.statology.org/equal-frequency-binning-python/
        # endpoint=False is used to prevent a x>xp endpoint repetition
        bins: ndarray = np.interp(
            x=np.linspace(0, len(x), nlattice + 1, endpoint=False),
            xp=np.arange(len(x)),
            fp=np.sort(x),
        )

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
            bins = _respace_bins(deepcopy(bins), maxsep, onsky=True, eps=2 * np.finfo(maxsep.dtype).eps)

        res = binned_statistic(x, data.T, bins=bins, statistic="median")
        prototypes: ndarray = res.statistic.T

        # When there is no data in a bin, it is set to NaN.
        # This is replaced with the interpolation from nearby points.
        for j, d in enumerate(prototypes.T):
            mask = np.isnan(d)
            prototypes[mask, j] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), d[~mask])

        return prototypes

    # ===============================================================

    # ---------------------------------------------------------------
    # fitting

    def _activation_distance(self, x: ndarray, w: ndarray) -> ndarray:
        """Activation distance.

        Parameters
        ----------
        x : (D,) ndarray
            Point.
        w : (L, D) ndarray
            Prototypes.

        Returns
        -------
        ndarray
        """
        # for the positions (lon, lat)
        pd: ndarray = angular_separation(*x[:2], *w.T[:2, :])

        # velocity distance
        if len(x) <= 2 * 2:
            vd = 0
        else:
            # FIXME! have to transform to cartesian and then back.
            _, xv = s2pv(theta=x[0], phi=x[1], r=1, td=x[2], pd=x[3], rd=0)
            _, wv = s2pv(theta=w[:, 0], phi=w[:, 1], r=1, td=w[:, 2], pd=w[:, 3], rd=0)
            vd = norm(subtract(xv, wv), axis=-1)

        return pd + vd

    def _update(self, x: ndarray, t: int, max_iteration: int) -> None:
        """Update the locations of the prototypes.

        Parameters
        ----------
        x : (D,) ndarray['radian']
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

        # The first two dimensions are angular. The distance should be on the sphere.
        ps = self.prototypes
        separation = angular_separation(ps[:, 0], ps[:, 1], x[0], x[1])
        posang = position_angle(ps[:, 0], ps[:, 1], x[0], x[1])
        newlon, newlat = offset_by(ps[:, 0], ps[:, 1], posang=posang, distance=g * separation)
        # keep in correct phase lon (-pi, pi), lat (-pi/2, pi/2)
        # TODO! check that can do these separately. Might need to do simultaneously.
        newlon = (newlon + pi) % twopi - pi
        newlat = (newlat + halfpi) % pi - halfpi
        self.prototypes[:, :2] = np.c_[newlon, newlat]  # assign on view (works around frozen)

        # TODO! better treatment of on-sphere
        if ps.shape[1] > 2:  # kinematics
            self.prototypes[:, 2:] += g[:, None] * (x[2:] - ps[:, 2:])

    # ---------------------------------------------------------------
    # predicting structure

    def predict(self, crd: ndarray, /) -> tuple[ndarray, ndarray]:
        """Order data from SOM in 2+N Dimensions.

        Parameters
        ----------
        crd : ndarray
            This will generally be the same data used to train the SOM.
        origin : ndarray or None

        Returns
        -------
        ndarray

        Notes
        -----
        The SOM creates a 1D lattice of connected nodes ($q$'s, gray) ordered by
        proximity to the designated origin, then along the lattice.
        Data points ($p$'s, green) are assigned an order from the SOM-lattice.
        The distance from the data to each node is computed. Likewise the projection
        is found of each data point on the edges connecting each SOM node.
        All projections lying outside the edges (shaded regions) are eliminated,
        also eliminated are all but the closest nodes. Remaining edges and node
        connections in dotted block, with projections labeled $e$.
        Data points are sorted into the closest node regions (blue)
        and edge regions (shaded).
        Data points in end-cap node regions are sorted by extended
        projection on nearest edge.
        Data points in edge regions are ordered by projection along the edge.
        Data points in intermediate node regions are ordered by the angle between
        the edge regions.
        """
        projdata, ordering = super().predict(crd)

        # Correct for possible phase wraps
        # TODO! more general correction for arbitrary number of phase wraps
        crd = crd[:, 0][ordering]
        discont = pi / 2  # [rad]
        jumps = np.where(np.diff(crd) >= discont)[0]
        if len(jumps) == 1:
            i = jumps[0] + 1
            ordering = np.concatenate((ordering[i:], ordering[:i]))

        projdata = projdata[ordering]

        return projdata, ordering
