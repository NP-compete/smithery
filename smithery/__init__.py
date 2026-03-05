"""Smithery: Forge tool-calling models from your API definitions."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("smithery")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = ["__version__"]
