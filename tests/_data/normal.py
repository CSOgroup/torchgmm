# pylint: disable=missing-function-docstring

import torch


def sample_data(counts: list[int], dims: list[int]) -> list[torch.Tensor]:
    return [torch.randn(count, dim) for count, dim in zip(counts, dims, strict=False)]


def sample_means(counts: list[int], dims: list[int]) -> list[torch.Tensor]:
    return [torch.randn(count, dim) for count, dim in zip(counts, dims, strict=False)]


def sample_spherical_covars(counts: list[int]) -> list[torch.Tensor]:
    return [torch.rand(count) for count in counts]


def sample_diag_covars(counts: list[int], dims: list[int]) -> list[torch.Tensor]:
    return [torch.rand(count, dim).squeeze() for count, dim in zip(counts, dims, strict=False)]


def sample_full_covars(counts: list[int], dims: list[int]) -> list[torch.Tensor]:
    result = []
    for count, dim in zip(counts, dims, strict=False):
        A = torch.rand(count, dim * 10, dim)
        result.append(A.permute(0, 2, 1).bmm(A).squeeze())
    return result
