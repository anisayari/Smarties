"""Smarties text classification and Wikipedia corpus tools."""

from . import SmartiesAPI, rake
from .classifier import TextClassifier

__all__ = ["TextClassifier", *SmartiesAPI.__all__]


def __getattr__(name):
    # Expose the historical package-level API, including the current language.
    if name in __all__:
        return getattr(SmartiesAPI, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
