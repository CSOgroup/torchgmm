from typing import Generic

from ._protocols import D_contra, Predictor, R_co, Transformer


class TransformerMixin(Generic[D_contra, R_co]):
    """
    Mixin that provides a ``fit_transform`` method that chains fitting the estimator and
    transforming the data it was fitted on.
    """

    def fit_transform(self: Transformer[D_contra, R_co], data: D_contra) -> R_co:
        """
        Fits the estimator using the provided data and subsequently transforms the data using the
        fitted estimator. It simply chains calls to :meth:`fit` and :meth:`transform`.

        Args:
            data: The data to use for fitting and to transform. The data must have the
                same type as for the :meth:`fit` method.

        Returns
        -------
            The transformed data. Consult the :meth:`transform` documentation for more information
            on the return type.
        """
        return self.fit(data).transform(data)


class PredictorMixin(Generic[D_contra, R_co]):
    """
    Mixin that provides a ``fit_predict`` method that chains fitting the estimator and making
    predictions for the data it was fitted on.
    """

    def fit_predict(self: Predictor[D_contra, R_co], data: D_contra) -> R_co:
        """
        Fits the estimator using the provided data and subsequently predicts the labels for the
        data using the fitted estimator. It simply chains calls to :meth:`fit` and
        :meth:`predict`.

        Args:
            data: The data to use for fitting and to predict labels for. The data must have the
                same type as for the :meth:`fit` method.

        Returns
        -------
            The predicted labels. Consult the :meth:`predict` documentation for more information
            on the return type.
        """
        return self.fit(data).predict(data)
