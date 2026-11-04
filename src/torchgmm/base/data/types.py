from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, TensorDataset

T = TypeVar("T")

# The aliases below are documented with attribute docstrings: union objects do not allow setting
# `__doc__` (on Python 3.14+ this also applies to `typing.Union`).
TensorLike = npt.NDArray[np.float32] | torch.Tensor
"""
Type annotation for functions accepting any kind of tensor data as input. Consider using this
annotation if your methods in an estimator derived from :class:`torchgmm.base.BaseEstimator` work on
tensors.
"""

DataLoaderLike = DataLoader[T] | Sequence[DataLoader[T]]
"""
Generic type annotation for functions accepting any data loader as input. Consider using this
annotation for the implementation of methods in an estimator derived from
:class:`torchgmm.base.BaseEstimator`.
"""


def dataset_from_tensors(*data: TensorLike) -> TensorDataset:
    """
    Transforms a set of tensor-like items into a datasets.

    Args:
        data: The tensor-like items.

    Returns
    -------
        The dataset.
    """
    return TensorDataset(*[_to_tensor(t) for t in data])


def _to_tensor(data: TensorLike) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    return data
