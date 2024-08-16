from abc import ABC, abstractmethod
from typing import List

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

        def training_epoch_end(self, outputs: List[torch.Tensor]) -> None:
            """Training epoch end hook for PyTorch Lightning < 2.0.0."""
            self.nonparametric_training_epoch_end()

    @abstractmethod
    def nonparametric_training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        """Training step that is not allowed to return any value."""

    def nonparametric_training_epoch_end(self) -> None:
        """
        Training epoch end that is not passed any outputs.

        Does nothing by default.
        """

    def all_gather_first(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gathers the provided tensor from all processes.

        If more than one process is available, chooses the value of the first process in every
        process.
        """
        gathered = self.all_gather(x)
        if gathered.dim() > x.dim():
            return gathered[0]
        return x
