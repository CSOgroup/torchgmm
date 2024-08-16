# pylint: disable=missing-class-docstring,missing-function-docstring

from typing import Protocol, TypeVar

D_contra = TypeVar("D_contra", contravariant=True)
R_co = TypeVar("R_co", covariant=True)
E = TypeVar("E", bound="Estimator")  # type: ignore


class Estimator(Protocol[D_contra]):
    def fit(self: E, data: D_contra) -> E: ...


class Transformer(Estimator[D_contra], Protocol[D_contra, R_co]):
    def transform(self, data: D_contra) -> R_co: ...


class Predictor(Estimator[D_contra], Protocol[D_contra, R_co]):
    def predict(self, data: D_contra) -> R_co: ...
