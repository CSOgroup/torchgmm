from collections.abc import Iterator
from typing import Any, TypeVar

try:
    from pytorch_lightning.overrides.distributed import _IndexBatchSamplerWrapper as IndexBatchSamplerWrapper
except ImportError:
    from pytorch_lightning.overrides.distributed import IndexBatchSamplerWrapper
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data.sampler import SequentialSampler

from .collation import collate_tuple
from .sampler import RangeBatchSampler

T_co = TypeVar("T_co", covariant=True)


class DataLoader(TorchDataLoader[T_co]):
    """Extension for PyTorch's builtin dataloader. This implementation allows
    to retrieve contiguous indices from a
    :class:`~torch.utils.data.TensorDataset` orders of magnitude faster. The
    data loader, thus, enables to implement traditional machine learning
    methods that exhibit a speed similar to the implementations found in
    Scikit-learn.

    Note:
        Retrieving contiguous indices is only possible when all of the following conditions apply:

        - ``shuffle=False`` or ``batch_sampler`` is of type
          :class:`~torchgmm.base.data.RangeBatchSampler`
        - ``sampler is None``
        - ``num_workers=0``
        - ``dataset`` is not iterable
    """

    def __init__(self, dataset: Dataset[T_co], **kwargs: Any):
        """
        Args:
            dataset: The dataset from which to load the data.
            kwargs: Keyword arguments passed to :meth:`torch.utils.data.DataLoader.__init__`.
        """
        if (
            not kwargs.get("shuffle", False)
            and "sampler" not in kwargs
            and "batch_sampler" not in kwargs
            and kwargs.get("num_workers", 0) == 0
            and isinstance(dataset, TensorDataset)
        ):
            kwargs["batch_sampler"] = RangeBatchSampler(
                SequentialSampler(dataset),
                batch_size=kwargs.get("batch_size", 1),
                drop_last=kwargs.get("drop_last", False),
            )
            kwargs.pop("batch_size", None)
            kwargs.pop("shuffle", None)
            kwargs.pop("drop_last", None)
            kwargs.setdefault("collate_fn", collate_tuple)

        super().__init__(dataset, **kwargs)  # type: ignore

        self.custom_batching = self.num_workers == 0 and (
            isinstance(self.batch_sampler, RangeBatchSampler)
            or (
                self.batch_sampler is not None
                and hasattr(self.batch_sampler, "sampler")
                and isinstance(self.batch_sampler.sampler, RangeBatchSampler)
            )
            or (
                isinstance(self.batch_sampler, IndexBatchSamplerWrapper)
                and isinstance(self.batch_sampler._batch_sampler, RangeBatchSampler)  # type: ignore
            )
        )

    def __iter__(self) -> Iterator[Any]:  # pylint: disable=inconsistent-return-statements
        if not self.custom_batching:
            yield from super().__iter__()
            return

        for indices in self.batch_sampler:
            if isinstance(indices, range):
                subscript = slice(indices.start, indices.stop)
                yield self.collate_fn(tuple(t[subscript] for t in self.dataset.tensors))
            else:
                yield self.collate_fn(tuple(t[indices] for t in self.dataset.tensors))
