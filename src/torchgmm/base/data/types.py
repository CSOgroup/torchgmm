from typing import Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, TensorDataset

T = TypeVar("T")

TensorLike = Union[
    npt.NDArray[np.float32],
    torch.Tensor,
]
TensorLike.__doc__ = """
Type annotation for functions accepting any kind of tensor data as input. Consider using this
annotation if your methods in an estimator derived from :class:`torchgmm.base.BaseEstimator` work on
tensors.
"""

DataLoaderLike = Union[
    DataLoader[T],
    Sequence[DataLoader[T]],
]
DataLoaderLike.__doc__ = """
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
