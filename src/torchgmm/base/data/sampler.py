import math
from typing import Iterator

from torch.utils.data import Sampler
from torch.utils.data.sampler import SequentialSampler


class RangeBatchSampler(Sampler[range]):
    """
    Sampler providing batches of contiguous indices.

    This sampler can be used with
    :class:`torchgmm.base.data.DataLoader` to provide significant speedups for tensor datasets.
    """

    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool = False):
        """
        Args:
            sampler: The sampler providing indices. Must be a sequential sampler. Note that the
                only purpose of this sampler is to determine its length.
            batch_size: The number of items to sample for each batch.
            drop_last: Whether to drop the last batch if ``num_items`` is not divisible by
                ``batch_size``.
        """
        assert isinstance(sampler, SequentialSampler), f"{self.__class__.__name__} only works with sequential samplers."

        super().__init__(None)
        self.dataset_size = len(sampler)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)

    def __iter__(self) -> Iterator[range]:
        for i in range(len(self)):
            sample = range(i * self.batch_size, (i + 1) * self.batch_size)
            yield sample
