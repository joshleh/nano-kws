"""nano-kws: edge-deployable keyword spotter.

A focused walk through the train -> quantize -> export -> benchmark
pipeline for an on-device audio classifier. See the project README for
the design narrative and the model card for intended use / limitations.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nano-kws")
except PackageNotFoundError:  # editable install before metadata is built
    __version__ = "0.1.0"

__all__ = ["__version__"]
