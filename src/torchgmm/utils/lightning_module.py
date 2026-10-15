from abc import ABC, abstractmethod
from typing import Any

import pytorch_lightning as pl
import torch
from packaging import version
from torch import nn


class NonparametricLightningModule(pl.LightningModule, ABC):
    """A lightning module which sets some defaults for training models with no parameters (i.e. only buffers that are optimized differently than via gradient descent)."""

    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

        # Required parameter to make DDP training work
        self.register_parameter("__ddp_dummy__", nn.Parameter(torch.empty(1)))

        # Information about the fitting procedure. These are stored as (persistent) buffers such
        # that they are available in the main process when training in spawned/forked processes.
        self.fit_num_epochs: torch.Tensor
        self.register_buffer("fit_num_epochs", torch.zeros((), dtype=torch.long))
        self.fit_stopped_early: torch.Tensor
        self.register_buffer("fit_stopped_early", torch.zeros((), dtype=torch.bool))

    def configure_optimizers(self) -> None:
        """Configure optimizers hook from PyTorch Lightning."""
        return None

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Training step hook from PyTorch Lightning."""
        self.nonparametric_training_step(batch, batch_idx)

    if version.parse(pl.__version__) >= version.parse("2.0.0"):

        def on_train_epoch_end(self) -> None:
            """Training epoch end hook for PyTorch Lightning >= 2.0.0."""
            self.nonparametric_training_epoch_end()
    else:

        def training_epoch_end(self, outputs: list[torch.Tensor]) -> None:
            """Training epoch end hook for PyTorch Lightning < 2.0.0."""
            self.nonparametric_training_epoch_end()

    def on_train_end(self) -> None:
        """Train end hook from PyTorch Lightning."""
        self.fit_num_epochs.fill_(self.current_epoch)
        self.fit_stopped_early.fill_(self.trainer.should_stop)

    @abstractmethod
    def nonparametric_training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Training step that is not allowed to return any value."""

    def nonparametric_training_epoch_end(self) -> None:
        """
        Training epoch end that is not passed any outputs.

        Does nothing by default.
        """

    def all_gather_list(self, x: torch.Tensor, group: Any | None = None) -> list[torch.Tensor]:
        """
        Gathers the provided tensor from all processes and returns one tensor per process.

        This function is meant to be used as ``dist_sync_fn`` of :mod:`torchmetrics` metrics,
        which expect a list of tensors (one per process) and pass a ``group`` argument.
        """
        gathered = self.all_gather(x, group=group)
        if gathered.dim() > x.dim():
            return list(gathered.unbind(0))
        return [x]

    def broadcast_first(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the value of the provided tensor in the first process in every process.

        This is required for values which are computed independently (e.g. randomly) on each
        process, but must be the same across processes.
        """
        # The tensor is moved to the CPU such that it is not received on the device of the first
        # process
        return self.trainer.strategy.broadcast(x.cpu(), src=0).to(x.device)
