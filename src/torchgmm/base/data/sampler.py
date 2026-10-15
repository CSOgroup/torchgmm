import math
from typing import Iterator

from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler


class RangeBatchSampler(Sampler[range]):
    """
    Sampler providing batches of contiguous indices.

    This sampler can be used with
    :class:`torchgmm.base.data.DataLoader` to provide significant speedups for tensor datasets.

    When used in a distributed setting (i.e. when PyTorch Lightning injects a
    :class:`~torch.utils.data.distributed.DistributedSampler`), every process receives a
    contiguous shard of the dataset. Unlike the distributed sampler, the dataset is not padded
    with repeated items, such that every item is seen exactly once per epoch. Nonetheless, all
    processes run the same number of (non-empty) batches, as required for distributed training.
    """

    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool = False):
        """
        Args:
            sampler: The sampler providing indices. Must be a sequential sampler or a
                distributed sampler (whose shuffling and padding are ignored). Note that the only
                purpose of this sampler is to determine the range of indices to iterate over.
            batch_size: The number of items to sample for each batch.
            drop_last: Whether to drop the last batch if ``num_items`` is not divisible by
                ``batch_size``.
        """
        # NOTE: `Sampler.__init__` is not called on purpose. It does nothing, but its signature
        # differs across PyTorch versions (`data_source` is required for some versions and was
        # removed in others).
        self.batch_size = batch_size
        self.drop_last = drop_last

        if isinstance(sampler, DistributedSampler):
            # PyTorch Lightning always enables shuffling for the distributed sampler it injects at
            # training time. It is ignored here since the order of the data is irrelevant for the
            # algorithms implemented in TorchGMM. However, the wrapped sampler must be sequential.
            wrapped = getattr(sampler.dataset, "_sampler", None)
            assert wrapped is None or isinstance(
                wrapped, SequentialSampler
            ), f"{self.__class__.__name__} only works with sequential samplers."

            # Split the dataset into contiguous shards whose sizes differ by at most one
            num_items = len(sampler.dataset)  # type: ignore
            base, remainder = divmod(num_items, sampler.num_replicas)
            self.offset = sampler.rank * base + min(sampler.rank, remainder)
            self.num_items = base + int(sampler.rank < remainder)
            max_num_items = base + int(remainder > 0)
            min_num_items = base
        else:
            assert isinstance(
                sampler, SequentialSampler
            ), f"{self.__class__.__name__} only works with sequential samplers."
            self.offset = 0
            self.num_items = len(sampler)
            max_num_items = min_num_items = self.num_items

        # The number of batches must be the same for all processes
        if drop_last:
            self.num_batches = min_num_items // batch_size
        else:
            self.num_batches = math.ceil(max_num_items / batch_size)
            if self.num_items < self.num_batches:
                raise ValueError(
                    f"Cannot split the {self.num_items} items of this process into {self.num_batches} "
                    f"non-empty batches of size at most {batch_size}. Increase the batch size or reduce "
                    "the number of processes."
                )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[range]:
        start = 0
        for i in range(self.num_batches):
            # Batches have the configured batch size, except that the last batches may be smaller
            # in order to ensure that none of the batches is empty.
            end = min((i + 1) * self.batch_size, self.num_items - (self.num_batches - i - 1))
            yield range(self.offset + start, self.offset + end)
            start = end
