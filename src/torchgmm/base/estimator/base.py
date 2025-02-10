from __future__ import annotations

import copy
import inspect
import json
import logging
import pickle
import warnings
from abc import ABC
from pathlib import Path
from typing import Any, TypeVar

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from torchgmm.base.utils.path import PathType

from .exception import NotFittedError

E = TypeVar("E", bound="BaseEstimator")  # type: ignore
T = TypeVar("T")

logger = logging.getLogger(__name__)


class BaseEstimator(ABC):
    """
    Base estimator class that all estimators should inherit from. This base estimator does not
    enforce the implementation of any methods, but users should follow the Scikit-learn guide on
    implementing estimators (which can be found `here <https://scikit-
    learn.org/stable/developers/develop.html>`_). Some of the methods mentioned in this guide are
    already implemented in this base estimator and work as expected if the aspects listed below are
    followed.

    In contrast to Scikit-learn's estimator, this estimator is strongly typed and integrates well
    with PyTorch Lightning. Most importantly, it provides the :meth:`trainer` method which returns
    a fully configured trainer to be used by other methods. The configuration is stored in the
    estimator and can be adjusted by passing parameters to ``default_params``, ``user_params`` and
    ``overwrite_params`` when calling ``super().__init__()``. By default, the base estimator sets
    the following flags:

    - Logging is disabled (``logger=False``).
    - Logging is performed at every step (``log_every_n_steps=1``).
    - The progress bar is only enabled (``enable_progress_bar``) if TorchGMM's logging level is
        ``INFO`` or more verbose.
    - Checkpointing is only enabled (``enable_checkpointing``) if TorchGMM's logging level is
        ``DEBUG`` or more verbose.
    - The model summary is only enabled (``enable_model_summary``) if TorchGMM's logging level is
        ``DEBUG`` or more verbose.

    Note that the logging level can be changed via :meth:`torchgmm.base.set_logging_level`.

    When subclassing this base estimator, users should take care of the following aspects:

    - All parameters passed to the initializer must be assigned to attributes with the same name.
      This ensures that :meth:`get_params` and :meth:`set_params` work as expected. Parameters that
      are passed to the trainer *must* be named ``trainer_params`` and should not be manually
      assigned to an attribute (this is handled by the base estimator).
    - Fitted attributes must (1) have a single trailing underscore (e.g. ``model_``) and (2) be
      defined as annotations. This ensures that :meth:`save` and :meth:`load` properly manage the
      estimator's persistence.
    """

    def __init__(
        self,
        *,
        default_params: dict[str, Any] | None = None,
        user_params: dict[str, Any] | None = None,
        overwrite_params: dict[str, Any] | None = None,
    ):
        """
        Args:
            default_params: Estimator-specific parameters that provide defaults for configuring the
                PyTorch Lightning trainer. An example might be setting ``max_epochs``. Overwrites
                the default parameters established by the base estimator.
            user_params: User-specific parameters that configure the PyTorch Lightning trainer.
                This dictionary should be passed through from a ``trainer_params`` init argument in
                subclasses. Overwrites any of the default parameters.
            overwrite_params: PyTorch Lightning trainer flags that need to be ensured independently
                of user-provided parameters. For example, ``max_epochs`` could be fixed to a
                certain value.
        """
        self.trainer_params_user = user_params
        self.trainer_params = {
            **{
                "logger": False,
                "log_every_n_steps": 1,
                "enable_progress_bar": logger.getEffectiveLevel() <= logging.INFO,
                "enable_checkpointing": logger.getEffectiveLevel() <= logging.DEBUG,
                "enable_model_summary": logger.getEffectiveLevel() <= logging.DEBUG,
            },
            **(default_params or {}),
            **(user_params or {}),
            **(overwrite_params or {}),
        }

    def trainer(self, **kwargs: Any) -> pl.Trainer:
        """
        Returns the trainer as configured by the estimator. Typically, this method is only called
        by functions in the estimator.

        Args:
            kwargs: Additional arguments that override the trainer arguments registered in the
                initializer of the estimator.

        Returns
        -------
            A fully initialized PyTorch Lightning trainer.

        Note:
            This function should be preferred over initializing the trainer directly. It ensures
            that the returned trainer correctly deals with TorchGMM components that may be
            introduced in the future.
        """
        return pl.Trainer(**{**self.trainer_params, **kwargs})

    # ---------------------------------------------------------------------------------------------
    # PERSISTENCE

    @property
    def persistent_attributes(self) -> list[str]:
        """
        Returns the list of fitted attributes that ought to be saved and loaded.

        By default, this encompasses all annotations.
        """
        return list(self.__annotations__.keys())

    def save(self, path: PathType) -> None:
        """Saves the estimator to the provided directory. It saves a file named
        ``estimator.pickle`` for the configuration of the estimator and
        additional files for the fitted model (if applicable). For more
        information on the files saved for the fitted model or for more
        customization, look at :meth:`get_params` and
        :meth:`torchgmm.base.nn.Configurable.save`.

        Args:
            path: The directory to which all files should be saved.

        Note:
            This method may be called regardless of whether the estimator has already been fitted.

        Attention:
            If the dictionary returned by :meth:`get_params` is not JSON-serializable, this method
            uses :mod:`pickle` which is not necessarily backwards-compatible.
        """
        path = Path(path)
        assert not path.exists() or path.is_dir(), "Estimators can only be saved to a directory."

        path.mkdir(parents=True, exist_ok=True)
        self.save_parameters(path)
        try:
            self.save_attributes(path)
        except NotFittedError:
            # In case attributes are not fitted, we just don't save them
            pass

    def save_parameters(self, path: Path) -> None:
        """
        Saves the parameters of this estimator. By default, it uses JSON and falls back to
        :mod:`pickle`. It subclasses use non-primitive types as parameters, they should overwrite
        this method.

        Typically, this method should not be called directly. It is called as part of :meth:`save`.

        Args:
            path: The directory to which the parameters should be saved.
        """
        params = self.get_params()
        try:
            data = json.dumps(params, indent=4)
            with (path / "params.json").open("w+") as f:
                f.write(data)
        except TypeError:
            warnings.warn(
                f"Failed to serialize parameters of `{self.__class__.__name__}` to JSON. Falling back to `pickle`.",
                stacklevel=2,
            )
            with (path / "params.pickle").open("wb+") as f:
                pickle.dump(params, f)

    def save_attributes(self, path: Path) -> None:
        """
        Saves the fitted attributes of this estimator. By default, it uses JSON and falls back to
        :mod:`pickle`. Subclasses should overwrite this method if non-primitive attributes are
        fitted.

        Typically, this method should not be called directly. It is called as part of :meth:`save`.

        Args:
            path: The directory to which the fitted attributed should be saved.

        Raises
        ------
            NotFittedError: If the estimator has not been fitted.
        """
        if len(self.persistent_attributes) == 0:
            return

        attributes = {attribute: getattr(self, attribute) for attribute in self.persistent_attributes}
        try:
            data = json.dumps(attributes, indent=4)
            with (path / "attributes.json").open("w+") as f:
                f.write(data)
        except TypeError:
            warnings.warn(
                f"Failed to serialize fitted attributes of `{self.__class__.__name__}` to JSON. "
                "Falling back to `pickle`.",
                stacklevel=2,
            )
            with (path / "attributes.pickle").open("wb+") as f:
                pickle.dump(attributes, f)

    @classmethod
    def load(cls: type[E], path: PathType) -> E:
        """
        Loads the estimator and (if available) the fitted model. This method should only be
        expected to work to load an estimator that has previously been saved via :meth:`save`.

        Args:
            path: The directory from which to load the estimator.

        Returns
        -------
            The loaded estimator, either fitted or not.
        """
        path = Path(path)
        assert path.is_dir(), "Estimators can only be loaded from a directory."

        estimator = cls.load_parameters(path)
        try:
            estimator.load_attributes(path)
        except FileNotFoundError:
            warnings.warn(f"Failed to read fitted attributes of `{cls.__name__}` at path '{path}'", stacklevel=2)

        return estimator

    @classmethod
    def load_parameters(cls: type[E], path: Path) -> E:
        """
        Initializes this estimator by loading its parameters. If subclasses overwrite
        :meth:`save_parameters`, this method should also be overwritten.

        Typically, this method should not be called directly. It is called as part of :meth:`load`.

        Args:
            path: The directory from which the parameters should be loaded.
        """
        json_path = path / "params.json"
        pickle_path = path / "params.pickle"

        if json_path.exists():
            with json_path.open() as f:
                params = json.load(f)
        else:
            with pickle_path.open("rb") as f:
                params = pickle.load(f)

        return cls(**params)

    def load_attributes(self, path: Path) -> None:
        """
        Loads the fitted attributes that are stored at the fitted path. If subclasses overwrite
        :meth:`save_attributes`, this method should also be overwritten.

        Typically, this method should not be called directly. It is called as part of :meth:`load`.

        Args:
            path: The directory from which the parameters should be loaded.

        Raises
        ------
            FileNotFoundError: If the no fitted attributes have been stored.
        """
        json_path = path / "attributes.json"
        pickle_path = path / "attributes.pickle"

        if json_path.exists():
            with json_path.open() as f:
                self.set_params(json.load(f))
        else:
            with pickle_path.open("rb") as f:
                self.set_params(pickle.load(f))

    # ---------------------------------------------------------------------------------------------
    # SKLEARN INTERFACE

    def get_params(self, deep: bool = True) -> dict[str, Any]:  # pylint: disable=unused-argument
        """
        Returns the estimator's parameters as passed to the initializer.

        Args:
            deep: Ignored. For Scikit-learn compatibility.

        Returns
        -------
            The mapping from init parameters to values.
        """
        signature = inspect.signature(self.__class__.__init__)
        parameters = [p.name for p in signature.parameters.values() if p.name != "self"]
        return {p: getattr(self, p) for p in parameters}

    def set_params(self: E, values: dict[str, Any]) -> E:
        """
        Sets the provided values on the estimator. The estimator is returned as well, but the
        estimator on which this function is called is also modified.

        Args:
            values: The values to set.

        Returns
        -------
            The estimator where the values have been set.
        """
        for key, value in values.items():
            setattr(self, key, value)
        return self

    def clone(self: E) -> E:
        """
        Clones the estimator without copying any fitted attributes. All parameters of this
        estimator are copied via :meth:`copy.deepcopy`.

        Returns
        -------
            The cloned estimator with the same parameters.
        """
        return self.__class__(
            **{
                name: param.clone() if isinstance(param, BaseEstimator) else copy.deepcopy(param)
                for name, param in self.get_params().items()
            }
        )

    # ---------------------------------------------------------------------------------------------
    # SPECIAL METHODS

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__:
            return self.__dict__[key]
        if key.endswith("_") and not key.endswith("__") and key in self.__annotations__:
            raise NotFittedError(f"`{self.__class__.__name__}` has not been fitted yet")
        raise AttributeError(f"Attribute `{key}` does not exist on type `{self.__class__.__name__}`.")

    # ---------------------------------------------------------------------------------------------
    # PRIVATE

    def _num_batches_per_epoch(self, loader: DataLoader[Any]) -> int:
        """Returns the number of batches that are run for the given data loader
        across all processes when using the trainer provided by the
        :meth:`trainer` method. If ``n`` processes run.

        ``k`` batches each, this method returns ``k * n``.
        """
        trainer = self.trainer()
        num_batches = len(loader)  # type: ignore
        kwargs = trainer.distributed_sampler_kwargs
        if kwargs is None:
            return num_batches
        return num_batches * kwargs.get("num_replicas", 1)
