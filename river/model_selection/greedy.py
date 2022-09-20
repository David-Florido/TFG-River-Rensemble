from copy import deepcopy
from typing import List

from river.base import Regressor
from river.metrics import MAE, RegressionMetric

from .base import ModelSelectionRegressor


class GreedyRegressor(ModelSelectionRegressor):
    """Greedy selection regressor.

    This selection method simply updates each model at each time step. The current best model is
    used to make predictions. It's greedy in the sense that updating each model can be costly. On
    the other hand, bandit-like algorithms are more temperate in that only update a subset of the
    models at each step.

    Parameters
    ----------
    models
        The models to select from.
    metric
        The metric that is used to measure the performance of each model.

    Examples
    --------

    >>> from river import datasets
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import model_selection
    >>> from river import optim
    >>> from river import preprocessing

    >>> models = [
    ...     linear_model.LinearRegression(optimizer=optim.SGD(lr=lr))
    ...     for lr in [1e-5, 1e-4, 1e-3, 1e-2]
    ... ]

    >>> dataset = datasets.TrumpApproval()
    >>> metric = metrics.MAE()
    >>> model = (
    ...     preprocessing.StandardScaler() |
    ...     model_selection.GreedyRegressor(models, metric)
    ... )

    >>> evaluate.progressive_val_score(dataset, model, metric)
    MAE: 1.35

    """

    def __init__(self, models: List[Regressor], metric: RegressionMetric = None):
        if metric is None:
            metric = MAE()
        super().__init__(models, metric)
        self.metrics = [deepcopy(metric) for _ in range(len(self))]
        self._best_model = self[0]
        self._best_metric = self.metrics[0]

    def learn_one(self, x, y):
        for model, metric in zip(self, self.metrics):
            y_pred = model.predict_one(x)
            metric.update(y, y_pred)
            model.learn_one(x, y)

            if metric.is_better_than(self._best_metric):
                self._best_model = model
                self._best_metric = metric

        return self

    @property
    def best_model(self):
        return self._best_model
