from __future__ import annotations

import pytest
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler

from torchgmm.base.data import RangeBatchSampler


def _indices(batch_sampler: RangeBatchSampler) -> list[int]:
    return [i for batch in batch_sampler for i in batch]


@pytest.mark.parametrize("num_items,batch_size", [(10, 3), (10, 10), (10, 20)])
def test_sequential(num_items: int, batch_size: int):
    batch_sampler = RangeBatchSampler(SequentialSampler(range(num_items)), batch_size)
    assert all(isinstance(batch, range) for batch in batch_sampler)
    assert _indices(batch_sampler) == list(range(num_items))


@pytest.mark.parametrize("num_items,batch_size", [(10, 3), (11, 3), (11, 100), (7, 1)])
@pytest.mark.parametrize("num_replicas", [2, 3])
@pytest.mark.parametrize("shuffle", [False, True])
def test_distributed(num_items: int, batch_size: int, num_replicas: int, shuffle: bool):
    batch_samplers = [
        RangeBatchSampler(
            DistributedSampler(range(num_items), num_replicas=num_replicas, rank=rank, shuffle=shuffle),
            batch_size,
        )
        for rank in range(num_replicas)
    ]

    # All processes must run the same number of batches
    assert len({len(s) for s in batch_samplers}) == 1
    assert all(len(list(s)) == len(s) for s in batch_samplers)

    # Processes receive contiguous shards which (apart from padding) cover the dataset in order
    indices = [i for s in batch_samplers for i in _indices(s)]
    assert indices[:num_items] == list(range(num_items))
    assert all(0 <= i < num_items for i in indices)


@pytest.mark.parametrize("num_items", [10, 11])
def test_distributed_unpadded(num_items: int):
    num_replicas = 3
    batch_samplers = []
    for rank in range(num_replicas):
        sampler = DistributedSampler(range(num_items), num_replicas=num_replicas, rank=rank, shuffle=False)
        # Emulate PyTorch Lightning's sampler used for prediction which does not pad the data
        sampler.num_samples = len(range(rank, num_items, num_replicas))
        sampler.total_size = num_items
        batch_samplers.append(RangeBatchSampler(sampler, 2))

    indices = [i for s in batch_samplers for i in _indices(s)]
    assert indices == list(range(num_items))
