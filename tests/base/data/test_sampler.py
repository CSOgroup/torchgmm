from __future__ import annotations

import pytest
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler

from torchgmm.base.data import RangeBatchSampler


def _batches(batch_sampler: RangeBatchSampler) -> list[list[int]]:
    return [list(batch) for batch in batch_sampler]


@pytest.mark.parametrize("num_items,batch_size", [(10, 3), (10, 10), (10, 20)])
@pytest.mark.parametrize("drop_last", [False, True])
def test_sequential(num_items: int, batch_size: int, drop_last: bool):
    batch_sampler = RangeBatchSampler(SequentialSampler(range(num_items)), batch_size, drop_last)
    batches = _batches(batch_sampler)
    assert len(batches) == len(batch_sampler)
    assert all(isinstance(batch, range) for batch in batch_sampler)
    expected = list(range(num_items))
    if drop_last:
        expected = expected[: num_items // batch_size * batch_size]
    assert [i for batch in batches for i in batch] == expected
    assert all(len(batch) == batch_size for batch in batches[:-1])


@pytest.mark.parametrize("num_items,batch_size", [(10, 3), (11, 3), (11, 5), (11, 100), (6, 1)])
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
    batches = [_batches(s) for s in batch_samplers]

    # All processes must run the same number of non-empty batches of at most the batch size
    assert len({len(s) for s in batch_samplers}) == 1
    assert all(len(b) == len(s) for b, s in zip(batches, batch_samplers))
    assert all(0 < len(batch) <= batch_size for b in batches for batch in b)

    # Processes receive contiguous shards which cover the dataset exactly once (no padding)
    assert [i for b in batches for batch in b for i in batch] == list(range(num_items))


@pytest.mark.parametrize("num_items", [10, 11])
def test_distributed_drop_last(num_items: int):
    batch_samplers = [
        RangeBatchSampler(DistributedSampler(range(num_items), num_replicas=2, rank=rank), 2, drop_last=True)
        for rank in range(2)
    ]
    assert [len(s) for s in batch_samplers] == [2, 2]
    assert all(len(batch) == 2 for s in batch_samplers for batch in s)


def test_distributed_too_small():
    # The second process has a single item which cannot be split into two non-empty batches
    sampler = DistributedSampler(range(3), num_replicas=2, rank=1)
    with pytest.raises(ValueError):
        RangeBatchSampler(sampler, 1)
