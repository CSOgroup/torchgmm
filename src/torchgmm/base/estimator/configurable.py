from pathlib import Path
from typing import Any, Generic, TypeVar

from torchgmm.base.nn._protocols import ConfigurableModule
from torchgmm.base.utils import get_generic_type

from .base import BaseEstimator
from .exception import NotFittedError

M = TypeVar("M", bound=ConfigurableModule)  # type: ignore


class ConfigurableBaseEstimator(BaseEstimator, Generic[M]):
    """
    Extension of the base estimator which allows to manage a single model that uses the
    :class:`torchgmm.base.nn.Configurable` mixin.
    """

    model_: M

    def save_attributes(self, path: Path) -> None:
        # First, store simple attributes
        super().save_attributes(path)

        # Then, store the model
        self.model_.save(path / "model")

    def load_attributes(self, path: Path) -> None:
        # First, load simple attributes
        super().load_attributes(path)

        # Then, load the model
        model_cls = get_generic_type(self.__class__, ConfigurableBaseEstimator)
        self.model_ = model_cls.load(path / "model")  # type: ignore

    def __getattr__(self, key: str) -> Any:
        try:
            return super().__getattr__(key)
        except AttributeError as e:
            if key.endswith("_") and not key.endswith("__") and not key.startswith("_"):
                raise NotFittedError(f"`{self.__class__.__name__}` has not been fitted yet") from e
            raise e
