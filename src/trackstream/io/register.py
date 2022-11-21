"""Register IO extensions for trackstream."""

from __future__ import annotations

# STDLIB
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypedDict

# THIRD PARTY
from astropy.io.registry import IORegistryError, UnifiedIORegistry
from importlib_metadata import EntryPoint, entry_points

if TYPE_CHECKING:
    # THIRD PARTY
    from typing_extensions import NotRequired, TypeGuard


__all__ = ["UnifiedIOEntryPointRegistrar"]


##############################################################################
# PARAMETERS


class IdentifyCallable(Protocol):
    """Callable that identifies a format."""

    def __call__(self, origin: str, format: str | None, /, *args: Any, **kwargs: Any) -> bool:  # noqa: A002
        """Identify a format."""
        ...


class FuncCallable(Protocol):  # TODO!
    """Callable that reads or writes a format."""

    def __call__(self, base: Any, /, *args: Any, **kwds: Any) -> Any:
        """Read or write a format."""
        ...


class EPDict(TypedDict):
    """Entry-point dict."""

    data_class: type
    registry: UnifiedIORegistry
    func: FuncCallable
    identify: NotRequired[IdentifyCallable]


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class UnifiedIOEntryPointRegistrar:
    """Unified I/O entry-point registrar."""

    _ep_keys: ClassVar[frozenset[str]] = frozenset(getattr(EPDict, "__annotations__", {}).keys())
    """Do not change."""

    data_class: type
    group: str
    which: str

    def run(self) -> None:
        """Run the registrar."""
        eps = entry_points().select(group=self.group)
        for ep in eps:
            self(ep)

    def __call__(self, entry_point: EntryPoint) -> None:
        """Register an entry-point."""
        name = entry_point.name

        # Load entrypoint
        try:
            value = entry_point.load()
        except Exception as e:
            # This stops the fitting from choking if an entry_point produces an error.
            warnings.warn(
                f"{type(e).__name__} error occurred in entry point {name!r} -- not registering.\n\t{e.args}",
                UserWarning,
                stacklevel=2,
            )
            return

        if not self.isvalidentrypoint(value, name):
            return

        # register in any of reader, writer, identifier
        # the name is prepended with the module name, unless they match.
        # the Stream class defaults to StreamArm
        regname = name.replace("_", ".")
        data_class = value["data_class"]
        registry = value["registry"]

        registration_func = getattr(registry, "register_" + self.which)
        try:
            registration_func(regname, data_class, value["func"])
        except IORegistryError as e:
            warnings.warn(str(e))

        if "identify" in value:
            registry.register_identifier(regname, data_class, value["identify"], force=True)

    def isvalidentrypoint(self, value: Any, name: str) -> TypeGuard[EPDict]:
        """Validate a loaded entry-point."""
        valid: bool = True
        msg: str = ""
        # check entrypoint is correct
        if not isinstance(value, dict):
            msg = "expected to be a dict"
            valid = False
        # make sure only have allowed keys
        elif not self._ep_keys.issuperset(value.keys()):
            msg = f"can only contain keys {self._ep_keys}"
            valid = False

        # check data_class:
        elif not inspect.isclass(data_cls := value.get("data_class")):
            msg = "field 'data_class' must be a class"
            valid = False
        elif not issubclass(data_cls, self.data_class):
            msg = f"field 'data_class' must be a {self.data_class} subclass"
            valid = False

        # Check registry:
        elif not isinstance(value.get("registry"), UnifiedIORegistry):
            msg = f"field 'registry' must be a instance of {UnifiedIORegistry!r}"
            valid = False

        # Check actual funcs
        elif not callable(value.get("func")):
            msg = "field 'identify' must be a callable"
            valid = False
        elif "identify" in value and not callable(value["identify"]):
            msg = "field 'identify' must be a callable"
            valid = False

        if not valid:
            warnings.warn(f"entry point {name} {msg} -- not registering", UserWarning, stacklevel=2)

        return valid
