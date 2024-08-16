from typing import Tuple

import torch


def collate_tuple(batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    """
    Collate a tuple of batch items by returning the input tuple.

    This is the default used by
    :class:`~torchgmm.base.data.DataLoader` when slices are cut from the underlying data source.
    """
    return batch


def collate_tensor(batch: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    Collates a tuple of batch items into the first tensor.

    Might be useful if only a single tensor is passed to
    :class:`~torchgmm.base.data.DataLoader`.
    """
    return batch[0]
