import math
from typing import Iterator, List, Union

from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler


class RangeBatchSampler(Sampler[Union[range, List[int]]]):
    """
    Sampler providing batches of contiguous indices.

    This sampler can be used with
    :class:`torchgmm.base.data.DataLoader` to provide significant speedups for tensor datasets.

    When used in a distributed setting (i.e. when PyTorch Lightning injects a
    :class:`~torch.utils.data.distributed.DistributedSampler`), every process receives a
    contiguous shard of the dataset which is then split into batches.
    """

    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool = False):
        """
        Args:
            sampler: The sampler providing indices. Must be a sequential sampler or a
                distributed sampler (whose shuffling is ignored). Note that the only purpose of this sampler is to
                determine the range of indices to iterate over.
            batch_size: The number of items to sample for each batch.
            drop_last: Whether to drop the last batch if ``num_items`` is not divisible by
                ``batch_size``.
        """
        super().__init__()
        if isinstance(sampler, DistributedSampler):
            # PyTorch Lightning always enables shuffling for the distributed sampler it injects at
            # training time. It is ignored here since the order of the data is irrelevant for the
            # algorithms implemented in TorchGMM. However, the wrapped sampler must be sequential.
            wrapped = getattr(sampler.dataset, "_sampler", None)
            assert wrapped is None or isinstance(
                wrapped, SequentialSampler
            ), f"{self.__class__.__name__} only works with sequential samplers."

            # Every process gets a contiguous shard of the (possibly padded) dataset. This
            # partitions the data in the same way as the distributed sampler, only contiguously.
            self.num_items = len(sampler.dataset)  # type: ignore
            base, remainder = divmod(sampler.total_size, sampler.num_replicas)
            self.offset = sampler.rank * base + min(sampler.rank, remainder)
            self.dataset_size = sampler.num_samples
        else:
            assert isinstance(
                sampler, SequentialSampler
            ), f"{self.__class__.__name__} only works with sequential samplers."
            self.num_items = len(sampler)
            self.offset = 0
            self.dataset_size = len(sampler)

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)

    def __iter__(self) -> Iterator[Union[range, List[int]]]:
        for i in range(len(self)):
            start = self.offset + i * self.batch_size
            end = self.offset + min((i + 1) * self.batch_size, self.dataset_size)
            if end <= self.num_items:
                yield range(start, end)
            else:
                # The distributed sampler pads the dataset by repeating items from the start
                yield [j % self.num_items for j in range(start, end)]
