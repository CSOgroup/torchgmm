# pylint: disable=missing-function-docstring
import json
import math
import subprocess
import sys
from typing import Any, Dict, Optional

import pytest
import torch

from torchgmm.bayes import GaussianMixture
from torchgmm.clustering import KMeans

# Forking is not supported on Windows and unsafe on macOS
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="requires fork-based multiprocessing")

# The distributed fit is run in a fresh interpreter since forking a process in which PyTorch has
# already run parallel CPU operations (e.g. in previous tests) can deadlock the forked processes.
SCRIPT = """
import json, sys, torch
from torchgmm.bayes import GaussianMixture
from torchgmm.clustering import KMeans

estimator_name, batch_size = sys.argv[1], json.loads(sys.argv[2])
generator = torch.Generator().manual_seed(0)
means = torch.tensor([[-5.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
# An odd number of datapoints such that the data cannot be split evenly across processes
data = torch.cat([torch.randn(1001, 2, generator=generator) + m for m in means])
params = {"accelerator": "cpu", "devices": 2, "strategy": "ddp_fork", "enable_progress_bar": False}

torch.manual_seed(0)
if estimator_name == "kmeans_init":
    # Each process receives one of the clusters. Running zero epochs only performs the k-means++
    # initialization.
    data = torch.cat([torch.randn(1000, 2, generator=generator) - 100, torch.randn(1000, 2, generator=generator) + 100])
    estimator = KMeans(2, trainer_params={**params, "max_epochs": 0}).fit(data)
    result = {"centers": estimator.model_.centroids.tolist(), "score": 0.0}
elif estimator_name == "kmeans":
    estimator = KMeans(3, batch_size=batch_size, trainer_params=params).fit(data)
    result = {"centers": estimator.model_.centroids.tolist(), "score": estimator.inertia_}
else:
    estimator = GaussianMixture(3, covariance_type="diag", batch_size=batch_size, trainer_params=params).fit(data)
    result = {"centers": estimator.model_.means.tolist(), "score": estimator.nll_}
result.update(converged=estimator.converged_, num_iter=estimator.num_iter_)
print("RESULT " + json.dumps(result))
"""


def _data() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    means = torch.tensor([[-5.0, 0.0], [5.0, 0.0], [0.0, 5.0]])
    return torch.cat([torch.randn(1001, 2, generator=generator) + m for m in means])


def _fit_distributed(estimator_name: str, batch_size: Optional[int]) -> Dict[str, Any]:
    output = subprocess.run(
        [sys.executable, "-c", SCRIPT, estimator_name, json.dumps(batch_size)],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    line = next(line for line in output.stdout.splitlines() if line.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


def _sorted(x: torch.Tensor) -> torch.Tensor:
    return x[x[:, 0].argsort()]


@pytest.mark.parametrize("batch_size", [None, 1000])
def test_kmeans_ddp_fork(batch_size: Optional[int]):
    torch.manual_seed(0)
    single = KMeans(3, batch_size=batch_size).fit(_data())
    distributed = _fit_distributed("kmeans", batch_size)

    assert distributed["converged"]
    assert distributed["num_iter"] > 0
    assert torch.allclose(_sorted(torch.tensor(distributed["centers"])), _sorted(single.model_.centroids), atol=1e-4)
    assert math.isclose(distributed["score"], single.inertia_, rel_tol=1e-4)


@pytest.mark.parametrize("batch_size", [None, 1000])
def test_gmm_ddp_fork(batch_size: Optional[int]):
    torch.manual_seed(0)
    single = GaussianMixture(3, covariance_type="diag", batch_size=batch_size).fit(_data())
    distributed = _fit_distributed("gmm", batch_size)

    assert distributed["converged"]
    assert distributed["num_iter"] > 0
    assert torch.allclose(_sorted(torch.tensor(distributed["centers"])), _sorted(single.model_.means), atol=1e-3)
    assert math.isclose(distributed["score"], single.nll_, rel_tol=1e-4)


def test_kmeans_init_ddp_fork():
    # The initial centroids must be chosen from the data of all processes
    distributed = _fit_distributed("kmeans_init", None)
    centers = _sorted(torch.tensor(distributed["centers"]))
    assert torch.allclose(centers, torch.tensor([[-100.0, -100.0], [100.0, 100.0]]), atol=5)
