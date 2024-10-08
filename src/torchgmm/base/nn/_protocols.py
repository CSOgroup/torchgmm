# pylint: disable=missing-class-docstring,missing-function-docstring
from typing import Generic, Iterator, OrderedDict, Protocol, Tuple, Type, TypeVar

import torch
from torch import nn

from torchgmm.base.utils import PathType

C_co = TypeVar("C_co", covariant=True)
M = TypeVar("M", bound="ConfigurableModule")  # type: ignore


class ConfigurableModule(Protocol, Generic[C_co]):
    @property
    def config(self) -> C_co: ...

    @classmethod
    def load(cls: Type[M], path: PathType) -> M: ...

    def save(self, path: PathType, compile_model: bool = False) -> None: ...

    def save_config(self, path: PathType) -> None: ...

    def named_children(self) -> Iterator[Tuple[str, nn.Module]]: ...

    def state_dict(self) -> OrderedDict[str, torch.Tensor]: ...

    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor]) -> None: ...
