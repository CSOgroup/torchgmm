from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Generic

import torch
from torch import jit, nn

from torchgmm.base.utils import PathType, get_generic_type

from ._protocols import C_co, ConfigurableModule, M


class Configurable(Generic[C_co]):
    """
    A mixin for any PyTorch module to extend it with storage capabilities.

    By passing a single
    configuration object to the initializer, this mixin allows the module to be extended with
    :meth:`save` and :meth:`load` methods. These methods allow to (1) save the model along with
    its configuration (i.e. architecture) and (2) to load the model without instantiating an
    instance of the class.
    """

    def __init__(self, config: C_co, *args: Any, **kwargs: Any):
        """
        Args:
            config: The configuration of the architecture.
            args: Positional arguments that ought to be passed to the superclass.
            kwargs: Keyword arguments that ought to be passed to the superclass.
        """
        assert dataclasses.is_dataclass(config), "Configuration is not a dataclass."
        assert isinstance(self, nn.Module), "Configurable mixin can only be applied to subclasses of `torch.nn.Module`."

        super().__init__(*args, **kwargs)
        self.config = config

    @jit.unused
    def save_config(self: ConfigurableModule[C_co], path: Path) -> None:
        """
        Saves only the module's configuration to a file named ``config.json`` in the specified
        directory. This method should not be called directly. It is called as part of :meth:`save`.

        Args:
            path: The directory to which to save the configuration and parameter files. The
                directory may or may not exist but no parent directories are created.
        """
        path.mkdir(parents=False, exist_ok=True)
        with (path / "config.json").open("w+") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=4)

    @jit.unused
    def save(self: ConfigurableModule[C_co], path: PathType, compile_model: bool = False) -> None:
        """
        Saves the module's configuration and parameters to files in the specified directory. It
        creates two files, namely ``config.json`` and ``parameters.pt`` which contain the
        configuration and parameters, respectively.

        Args:
            path: The directory to which to save the configuration and parameter files. The
                directory may or may not exist but no parent directories are created.
            compile_model: Whether the model should be compiled via TorchScript. An additional file
                called ``model.ptc`` will then be stored. Note that you can simply load the
                compiled model via :meth:`torch.jit.load` at a later point.
        """
        path = Path(path)
        assert not path.exists() or path.is_dir(), "Modules can only be saved to a directory."

        path.mkdir(parents=True, exist_ok=True)

        # Store the model's configuration and all parameters
        self.save_config(path)
        with (path / "parameters.pt").open("wb+") as f:
            torch.save(self.state_dict(), f)  # pylint: disable=no-member

        # Optionally store the compiled model
        if compile_model:
            compiled_model = jit.script(self)
            with (path / "model.ptc").open("wb+") as f:
                jit.save(compiled_model, f)

    @classmethod
    def load_config(cls: type[M], path: Path) -> M:
        """
        Loads the module by reading the configuration. Parameters are initialized randomly as if
        the module would be initialized from scratch. This method should not be called directly. It
        is called as part of :meth:`load`.

        Args:
            path: The directory which contains the ``config.json`` to load.

        Returns
        -------
            The loaded model.

        Attention:
            This method must only be called if the module is initializable solely from a
            configuration.
        """
        config_cls = get_generic_type(cls, Configurable)
        with (path / "config.json").open("r") as f:
            config_args = json.load(f)
            config = _init_config(config_cls, config_args)

        return cls(config)  # type: ignore

    @classmethod
    def load(cls: type[M], path: PathType) -> M:
        """
        Loads the module's configurations and parameters from files in the specified directory at
        first. Then, it initializes the model with the stored configurations and loads the
        parameters. This method is typically used after calling :meth:`save` on the model.

        Args:
            path: The directory which contains the ``config.json`` and ``parameters.pt`` files to
                load.

        Returns
        -------
            The loaded model.

        Note:
            You can load modules even after you changed their configuration class. The only
            requirement is that any new configuration options have a default value.
        """
        path = Path(path)
        assert path.is_dir(), "Modules can only be loaded from a directory."

        # Load the config
        config_cls = get_generic_type(cls, Configurable)
        with (path / "config.json").open("r") as f:
            config_args = json.load(f)
            config = _init_config(config_cls, config_args)

        # Initialize model
        model = cls(config)  # type: ignore
        with (path / "parameters.pt").open("rb") as f:
            state_dict = torch.load(f, weights_only=True)
        model.load_state_dict(state_dict)  # pylint: disable=no-member
        return model

    def clone(self: M, copy_parameters: bool = True) -> M:
        """
        Clones this module by initializing another module with the same configuration.

        Args:
            copy_parameters: Whether to copy this module's parameters or initialize the new module
                with random parameters.

        Returns
        -------
            The cloned module.
        """
        cloned = self.__class__(self.config)  # type: ignore
        if copy_parameters:
            cloned.load_state_dict(self.state_dict())  # pylint: disable=no-member
        return cloned


def _init_config(target: type[Any], args: dict[str, Any]) -> Any:
    result = {}
    for key, val in args.items():
        arg_type = target.__dataclass_fields__[key].type  # type: ignore
        if dataclasses.is_dataclass(arg_type):
            result[key] = _init_config(arg_type, val)
        else:
            result[key] = val
    return target(**result)
