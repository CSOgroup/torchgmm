import logging

from .estimator import BaseEstimator, ConfigurableBaseEstimator

# This is taken from PyTorch Lightning and ensures that logging for this package is enabled
_root_logger = logging.getLogger()
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
if not _root_logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler())
    _logger.propagate = False


def set_logging_level(level: int) -> None:
    """
    Enables or disables logging for the entire module. By default, logging is enabled.

    Args:
        enabled: Whether to enable logging.
    """
    _logger.setLevel(level)


__all__ = ["BaseEstimator", "ConfigurableBaseEstimator"]
