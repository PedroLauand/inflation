"""Standardized ring-distribution classes."""

from importlib import import_module

from .ejm import EJMDistribution
from .ghz import GHZDistribution
from .nsi_pr import NSIPRDistribution
from .protocols import RingDistributionProtocol
from .rgb import RGBDistribution

__all__ = [
    "RingDistributionProtocol",
    "EJMDistribution",
    "GHZDistribution",
    "RGBDistribution",
    "NSIPRDistribution",
    "ValidationFailure",
    "ValidationReport",
    "validate_consistency_upto",
    "validate_normalization",
    "validate_factorization",
]


def __getattr__(name):
    if name in {
        "ValidationFailure",
        "ValidationReport",
        "validate_consistency_upto",
        "validate_normalization",
        "validate_factorization",
    }:
        validation = import_module(".validation", __name__)
        return getattr(validation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
