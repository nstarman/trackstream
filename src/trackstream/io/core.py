"""I/O."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from astropy.io.registry import UnifiedIORegistry, UnifiedReadWrite

__all__: list[str] = []
__doctest_skip__ = ["*"]

if TYPE_CHECKING:
    from trackstream.stream.core import StreamArm


##############################################################################
# CODE
##############################################################################


# ==============================================================================
# Read / Write

readwrite_registry = UnifiedIORegistry()


class StreamArmRead(UnifiedReadWrite):
    """Read and parse data to a `~stream.StreamArm`.

    This function provides the StreamArm interface to the Astropy unified I/O
    layer. This allows easily reading a file in supported data formats using
    syntax such as::

        >>> from trackstream.stream import StreamArm
        >>> arm1 = StreamArm.read('<file name>')

    When the ``read`` method is called from a subclass the subclass will
    provide a keyword argument ``Stream=<class>`` to the registered read
    method. The method uses this Stream class, regardless of the class
    indicated in the file, and sets parameters' default values from the class'
    signature.

    Get help on the available readers using the ``help()`` method::

      >>> StreamArm.read.help()  # Get help reading and list supported formats
      >>> StreamArm.read.help(format='<format>')  # Get detailed help on a format
      >>> StreamArm.read.list_formats()  # Print list of available formats

    See also: https://docs.astropy.org/en/stable/io/unified.html

    .. note::

        :meth:`~stream.StreamArm.read` and
        :meth:`~stream.StreamArm.from_format` currently access the
        same registry. This will be deprecated and formats intended for
        ``from_format`` should not be used here. Use ``StreamArm.read.help()``
        to confirm that the format may be used to read a file.

    Parameters
    ----------
    *args
        Positional arguments passed through to data reader. If supplied the
        first argument is typically the input filename.
    format : str (optional, keyword-only)
        File format specifier.
    **kwargs
        Keyword arguments passed through to data reader.

    Returns
    -------
    out : `~stream.StreamArm` subclass instance
        `~stream.StreamArm` corresponding to file contents.

    Notes
    -----

    """

    def __init__(self, stream: StreamArm, stream_cls: type[StreamArm]) -> None:
        super().__init__(stream, stream_cls, "read", registry=readwrite_registry)

    def __call__(self, *args: Any, **kwargs: Any) -> StreamArm:
        """Read and parse data to a `~stream.StreamArm`."""
        # LOCAL
        from trackstream import StreamArm

        # so subclasses can override, also pass the class as a kwarg.
        if self._cls is not StreamArm:
            kwargs.setdefault("Stream", self._cls)  # set, if not present
            # check that it is the correct class.
            valid = (self._cls,)
            if kwargs["Stream"] not in valid:
                msg = "keyword argument `Stream` must be the class."
                raise ValueError(msg)

        arm = self.registry.read(self._cls, *args, **kwargs)

        if not isinstance(arm, StreamArm):
            msg = "arm is not an instance of StreamArm"
            raise TypeError(msg)

        return arm


class StreamArmWrite(UnifiedReadWrite):
    """Write this StreamArm object out in the specified format.

    This function provides the StreamArm interface to the astropy unified I/O
    layer. This allows easily writing a file in supported data formats.

    Get help on the available writers for ``StreamArm`` using the ``help()``
    method::

      >>> StreamArm.write.help()  # Get help writing and list supported formats
      >>> StreamArm.write.help(format='<format>')  # Get detailed help on format
      >>> StreamArm.write.list_formats()  # Print list of available formats

    .. note::

        :meth:`~stream.StreamArm.write` and
        :meth:`~stream.StreamArm.to_format` currently access the
        same registry. This will be deprecated and formats intended for
        ``to_format`` should not be used here. Use ``StreamArm.write.help()``
        to confirm that the format may be used to write to a file.

    Parameters
    ----------
    *args
        Positional arguments passed through to data writer. If supplied the
        first argument is the output filename.
    format : str (optional, keyword-only)
        File format specifier.
    **kwargs
        Keyword arguments passed through to data writer.

    Notes
    -----

    """

    def __init__(self, stream: StreamArm, stream_cls: type[StreamArm]) -> None:
        super().__init__(stream, stream_cls, "write", registry=readwrite_registry)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Write this StreamArm object out in the specified format."""
        self.registry.write(self._instance, *args, **kwargs)


# ==============================================================================
# Format Interchange
# for transforming instances, e.g. StreamArm <-> dict

convert_registry = UnifiedIORegistry()


class StreamArmFromFormat(UnifiedReadWrite):
    """Transform object to a `~stream.StreamArm`.

    This function provides the StreamArm interface to the Astropy unified I/O
    layer. This allows easily parsing supported data formats using
    syntax such as::

      >>> from trackstream.stream import StreamArm
      >>> arm1 = StreamArm.from_format(table, format='Table')

    When the ``from_format`` method is called from a subclass the subclass will
    provide a keyword argument ``Stream=<class>`` to the registered parser.
    The method uses this stream class, regardless of the class indicated in
    the data, and sets parameters' default values from the class' signature.

    Get help on the available readers using the ``help()`` method::

      >>> StreamArm.from_format.help()  # Get help and list supported formats
      >>> StreamArm.from_format.help('<format>')  # Get detailed help on a format
      >>> StreamArm.from_format.list_formats()  # Print list of available formats

    See also: https://docs.astropy.org/en/stable/io/unified.html

    .. note::

        :meth:`~stream.StreamArm.from_format` and
        :meth:`~stream.StreamArm.read` currently access the
        same registry. This will be deprecated and formats intended for
        ``read`` should not be used here. Use ``StreamArm.to_format.help()``
        to confirm that the format may be used to convert to a StreamArm.

    Parameters
    ----------
    obj : object
        The object to parse according to 'format'
    *args
        Positional arguments passed through to data parser.
    format : str (optional, keyword-only)
        Object format specifier.
    **kwargs
        Keyword arguments passed through to data parser.

    Returns
    -------
    out : `~stream.StreamArm` subclass instance
        `~stream.StreamArm` corresponding to ``obj`` contents.

    """

    def __init__(self, instance: StreamArm, stream_cls: type[StreamArm]) -> None:
        super().__init__(instance, stream_cls, "read", registry=convert_registry)

    def __call__(self, obj: Any, *args: Any, **kwargs: Any) -> StreamArm:
        """Transform object to a `~stream.StreamArm`."""
        # LOCAL
        from trackstream import StreamArm

        # so subclasses can override, also pass the class as a kwarg.
        # allows for `StreamArm.read` and
        # `StreamArm.read(..., Stream=...)`
        kwargs.setdefault("Stream", self._cls)
        if self._cls is not StreamArm:
            # check that it is the correct stream, can be wrong if user
            # passes in e.g. `Stream.read(..., Stream=...)`
            valid = (self._cls,)
            if kwargs["Stream"] not in valid:
                msg = "keyword argument `Stream` must be the class."
                raise ValueError(msg)

        arm = self.registry.read(self._cls, obj, *args, **kwargs)

        if not isinstance(arm, StreamArm):
            raise OSError

        return arm


class StreamArmToFormat(UnifiedReadWrite):
    """Transform this StreamArm to another format.

    This function provides the StreamArm interface to the astropy unified I/O
    layer. This allows easily transforming to supported data formats.

    Get help on the available representations for ``StreamArm`` using the
    ``help()`` method::

      >>> StreamArm.to_format.help()  # Get help and list supported formats
      >>> StreamArm.to_format.help('<format>')  # Get detailed help on format
      >>> StreamArm.to_format.list_formats()  # Print list of available formats

    .. note::

        :meth:`~stream.StreamArm.to_format` and
        :meth:`~stream.StreamArm.write` currently access the same
        registry. This will be deprecated and formats intended for ``write``
        should not be used here. Use ``StreamArm.to_format.help()`` to confirm
        that the format may be used to convert a StreamArm.

    Parameters
    ----------
    format : str
        Format specifier.
    *args
        Positional arguments passed through to data writer. If supplied the
        first argument is the output filename.
    **kwargs
        Keyword arguments passed through to data writer.

    """

    def __init__(self, stream: StreamArm, stream_cls: type[StreamArm]) -> None:
        super().__init__(stream, stream_cls, "write", registry=convert_registry)

    def __call__(self, format: str, *args: Any, **kwargs: Any) -> Any:  # noqa: A002
        """Transform this StreamArm to another format."""
        return self.registry.write(self._instance, None, *args, format=format, **kwargs)
