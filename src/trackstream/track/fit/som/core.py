from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import Any, Literal, NoReturn, final

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u
import numpy as np
from matplotlib.pyplot import Axes
from numpy.random import Generator

# LOCAL
from trackstream.stream.core import StreamArm
from trackstream.stream.plural import StreamArmsBase
from trackstream.track.fit.som.base import SOM1DBase, SOMInfo
from trackstream.track.fit.som.cartesian import CartesianSOM
from trackstream.track.fit.som.sphere import USphereSOM
from trackstream.track.fit.som.utils import _get_info_for_projection
from trackstream.track.fit.utils import _c2v, _v2c
from trackstream.track.width.core import LENGTH, SPEED
from trackstream.track.width.oned.core import (
    AngularDiffWidth,
    AngularWidth,
    Cartesian1DiffWidth,
    Cartesian1DWidth,
)
from trackstream.track.width.plural import Widths
from trackstream.utils.visualization import CommonPlotDescriptorBase, DKindT

__all__: list[str] = []


#####################################################################


@dataclass(frozen=True)
class SOMPlotDescriptor(CommonPlotDescriptorBase["SelfOrganizingMap"]):
    def current(
        self,
        kind: DKindT = "positions",
        *,
        origin: bool = True,
        connect: bool = True,
        x_offset: u.Quantity | Literal[0] = 0,
        y_offset: u.Quantity | Literal[0] = 0,
        ax: Axes | None = None,
        format_ax: bool = False,
    ) -> Axes:
        som, _ax, _ = self._setup(ax=ax)
        (x, xn), (y, yn) = self._get_xy(som.prototypes, kind=kind)

        if connect:
            _ax.plot(x + x_offset, y + y_offset, c="k")
        _ax.scatter(x + x_offset, y + y_offset, marker="P", edgecolors="black", facecolor="none")
        if origin:
            self._origin(frame=som.frame, kind=kind, ax=_ax)

        if format_ax:  # Axes settings
            self._format_ax(_ax, frame=som.frame.name, x=xn, y=yn)
        return _ax

    def initial(
        self,
        kind: DKindT = "positions",
        *,
        origin: bool = True,
        connect: bool = True,
        x_offset: u.Quantity | Literal[0] = 0,
        y_offset: u.Quantity | Literal[0] = 0,
        ax: Axes | None = None,
        format_ax: bool = False,
    ) -> Axes:
        som, _ax, _ = self._setup(ax=ax)
        (x, xn), (y, yn) = self._get_xy(som.init_prototypes, kind=kind)

        if connect:
            _ax.plot(x + x_offset, y + y_offset, c="k")
        _ax.scatter(x + x_offset, y + y_offset, marker="P", edgecolors="black", facecolor="none")
        if origin:
            self._origin(frame=som.frame, kind=kind, ax=_ax)

        if format_ax:  # Axes settings
            self._format_ax(_ax, frame=som.frame.name, x=xn, y=yn)
        return _ax

    def __call__(
        self,
        kind: DKindT = "positions",
        *,
        origin: bool = True,
        connect: bool = True,
        initial_prototypes: bool = False,
        x_offset: u.Quantity | Literal[0] = 0,
        y_offset: u.Quantity | Literal[0] = 0,
        ax: Axes | None = None,
        format_ax: bool = False,
    ) -> Axes:
        if initial_prototypes:
            _ax = self.initial(kind=kind, origin=False, connect=False, x_offset=0, y_offset=0, ax=ax, format_ax=False)
        else:
            _ax = ax
        axes = self.current(
            kind=kind, origin=origin, connect=connect, x_offset=x_offset, y_offset=y_offset, ax=_ax, format_ax=format_ax
        )
        return axes


#####################################################################


@final
@dataclass(frozen=True)
class SelfOrganizingMap:

    plot = SOMPlotDescriptor()

    # ===============================================================

    som: SOM1DBase
    # frame info
    frame: coords.BaseCoordinateFrame
    origin: coords.SkyCoord

    @property
    def info(self) -> SOMInfo:
        return self.som.info

    # ===============================================================

    @singledispatchmethod
    @classmethod
    def from_format(
        cls,
        arm: object,
        /,
        onsky: bool,
        *,
        nlattice: int | None = None,
        sigma: float = 0.1,
        learning_rate: float = 0.3,
        rng: Generator | int | None = None,
        prototype_kw: dict[str, Any] = {},
    ) -> NoReturn:
        raise NotImplementedError("not dispatched")

    @from_format.register(StreamArm)
    @classmethod
    def _from_format_streamarm(
        cls,
        arm: StreamArm,
        /,
        onsky: bool,
        kinematics: bool | None = None,
        *,
        nlattice: int | None = None,
        sigma: float = 0.1,
        learning_rate: float = 0.3,
        rng: Generator | int | None = None,
        prototype_kw: dict[str, Any] = {},
    ) -> SelfOrganizingMap:
        if arm.frame is None:
            # LOCAL
            from trackstream.stream.base import FRAME_NONE_ERR

            raise FRAME_NONE_ERR

        SOM = USphereSOM if onsky else CartesianSOM

        som = SOM.from_format(
            arm,
            nlattice=nlattice,
            sigma=sigma,
            learning_rate=learning_rate,
            rng=rng,
            prototype_kw=prototype_kw,
            kinematics=kinematics,
        )

        return cls(som=som, frame=arm.frame, origin=arm.origin)

    @from_format.register(StreamArmsBase)
    @classmethod
    def _from_format_streamarmsbase(
        cls,
        arms: StreamArmsBase,
        /,
        onsky: bool,
        kinematics: bool | None = None,
        *,
        nlattice: int | None = None,
        sigma: float = 0.1,
        learning_rate: float = 0.3,
        rng: Generator | int | None = None,
        prototype_kw: dict[str, Any] = {},
    ) -> dict[str, SOM1DBase]:
        out: dict[str, SOM1DBase] = {}
        for k, arm in arms.items():
            out[k] = cls.from_format(
                arm,
                onsky=onsky,
                kinematics=kinematics,
                nlattice=nlattice,
                sigma=sigma,
                learning_rate=learning_rate,
                rng=rng,
                prototype_kw=prototype_kw,
            )
        return out

    # ===============================================================

    @property
    def nlattice(self) -> int:
        return self.som.nlattice

    @property
    def nfeature(self) -> int:
        return self.som.nfeature

    @property
    def prototypes(self) -> coords.BaseCoordinateFrame:
        """Read-only view of prototypes vectors as a |Frame|."""
        p = self.som.prototypes.view()
        p.flags.writeable = False
        return _v2c(self, p)

    @property
    def init_prototypes(self) -> coords.BaseCoordinateFrame:
        """Read-only view of prototypes vectors as a |Frame|."""
        p = self.som.init_prototypes.view()
        p.flags.writeable = False
        return _v2c(self, p)

    @property
    def onsky(self) -> bool:
        return True if isinstance(self.som, USphereSOM) else False

    @property
    def kinematics(self) -> bool:
        # two options for number of features
        soms = (CartesianSOM, USphereSOM)
        nf = (6, 4)
        i = soms.index(type(self.som))
        return True if self.som.nfeature == nf[i] else False

    # ===============================================================

    def fit(
        self,
        data: coords.SkyCoord,
        num_iteration: int = int(1e5),
        random_order: bool = False,
        progress: bool = True,
    ) -> None:
        v = _c2v(self, data)
        return self.som.fit(v, num_iteration=num_iteration, random_order=random_order, progress=progress)

    def predict(self, data: coords.SkyCoord, /) -> tuple[coords.SkyCoord, np.ndarray]:
        v = _c2v(self, data)
        projv, ordering = self.som.predict(v)

        # FIXME! add in the origin stuff

        projdata = coords.SkyCoord(_v2c(self, projv), copy=False)

        return projdata, ordering

    def fit_predict(
        self,
        data: coords.SkyCoord,
        /,
        num_iteration: int = int(1e5),
        random_order: bool = False,
        progress: bool = True,
        split: int | None = None,
    ) -> tuple[coords.SkyCoord, np.ndarray]:
        self.fit(data, num_iteration=num_iteration, random_order=random_order, progress=progress)
        projdata, order = self.predict(data)
        return projdata, order

    # ===============================================================

    def separation(self, data: coords.SkyCoord, /) -> Widths:
        """Compute orthogonal distances to SOM.

        Parameters
        ----------
        data : |SkyCoord|, positional-only

        Returns
        -------
        Widths
            With keys ``length`` and ``speed``.
        """
        # Prep
        qp = _c2v(self, data)
        D = qp.shape[1]
        units = tuple(self.info.units.values())  # from _c2v

        # Positions:
        iq = slice(0, D) if not self.kinematics else slice(0, D // 2)
        _, _, _, distances = _get_info_for_projection(qp[:, iq], self.som.prototypes[:, iq])
        ind_best_distance = np.argmin(distances, axis=1)
        orth_distance = distances[(np.arange(len(distances))), (ind_best_distance)]
        wcls = AngularWidth if self.onsky else Cartesian1DWidth
        qw = wcls(orth_distance * units[0][0])

        # Velocities:
        wdcls = AngularDiffWidth if self.onsky else Cartesian1DiffWidth
        ip = slice(D, None) if not self.kinematics else slice(D // 2, None)
        if not self.kinematics:
            pw = wdcls(np.full(len(qw), np.nan) * units[1][0])
        else:
            _, _, _, distances = _get_info_for_projection(qp[:, ip], self.som.prototypes[:, ip])

            ind_best_distance = np.argmin(distances, axis=1)
            orth_distance = distances[(np.arange(len(distances))), (ind_best_distance)]
            pw = wdcls(orth_distance * units[1][0])

        return Widths({LENGTH: qw, SPEED: pw})
