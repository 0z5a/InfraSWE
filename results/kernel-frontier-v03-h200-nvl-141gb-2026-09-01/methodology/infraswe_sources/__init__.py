"""InfraSWE executable benchmark harness."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("infraswe")
except PackageNotFoundError:  # source tree without an installed distribution
    __version__ = "0.1.0"

__all__ = ["__version__"]
