# pylint: disable=missing-function-docstring
import torch

from torchgmm.clustering.kmeans.metrics import DistanceSampler


def _gathered_distance_sampler(cumulative_squared_distances: list[float]) -> DistanceSampler:
    # Emulates the state of the sampler after gathering from multiple processes
    num_processes = len(cumulative_squared_distances)
    sampler = DistanceSampler(num_choices=3, num_features=2)
    sampler.choices = torch.arange(num_processes, dtype=torch.float).repeat_interleave(6).view(num_processes, 3, 2)
    sampler.cumulative_squared_distance = torch.tensor(cumulative_squared_distances).view(num_processes, 1)
    return sampler


def test_distance_sampler_gathered_zero_distances():
    # All distances are zero if all datapoints equal the existing centroid
    choices = _gathered_distance_sampler([0.0, 0.0]).compute()
    assert choices.shape == (3, 2)
    assert set(choices[:, 0].tolist()) <= {0.0, 1.0}


def test_distance_sampler_gathered_weights():
    # Choices are only sampled from the process holding datapoints far from existing centroids
    choices = _gathered_distance_sampler([0.0, 10.0]).compute()
    assert torch.equal(choices, torch.ones(3, 2))
