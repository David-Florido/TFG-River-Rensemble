import copy
import statistics
import collections
import pandas as pd
import numpy as np
import numpy.random as rnd
from typing import Counter
from river import ensemble
from river import preprocessing
from river import drift
from river import base
from river.base import drift_detector
from river.drift import ADWIN

__all__ = [
    "BaseRensembler",
    "RensemblerClassifier",
]

class BaseRensembler():

    def __init__(self, module_tuples, cfc: base.Classifier,  lam: float = 6.0, seed=None, drift_check = "off", unanimity_check: float = 0.0):
        self.modules = {}
        self.drift_detectors = {}
        self.cfc = {}
        self.lam = lam
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        self.unanimity_check = unanimity_check
        self.drift_check = drift_check

        if self.unanimity_check:
            self.n_unanimities_detected = 0
        
        module_cont = 0
        for module_tuple in module_tuples:
            n_modules = module_tuple[0]
            model_tuple = module_tuple[1]
            for n_module in range(n_modules):
                n_models = model_tuple[1]
                model = model_tuple[0]
                self.modules[module_cont] =  [copy.deepcopy(model) for _ in range(n_models)]
                if self.drift_check != "off":
                    self.n_drifts_detected = 0
                    self.drift_detectors[module_cont] = [copy.deepcopy(ADWIN()) for _ in range(n_models)]
                self.cfc[module_cont] = copy.deepcopy(cfc)
                module_cont += 1
    
    def learn_one(self, x, y):
        for module in self.modules:
            models = self.modules[module]
            predictions = {}
            change_detected = False
            model_cont = 0
            n_not_learnt = 0
            for model in models:
                learnt = False
                for k in range(self._rng.poisson(self.lam)):
                    learnt = True     
                    model.learn_one(x, y)

                y_pred = model.predict_one(x)

                if y_pred == None:
                    model.learn_one(x, y)
                    y_pred = model.predict_one(x)
                predictions[f'prediction_{model_cont}'] = y_pred
                    
                if self.drift_check != "off" and learnt != False:
                    try:
                        error_estimation = self.drift_detectors[module][model_cont].estimation
                        self.drift_detectors[module][model_cont].update(int(y_pred == y))
                        if self.drift_detectors[module][model_cont].change_detected:
                            if self.drift_detectors[module][model_cont].estimation > error_estimation:
                                change_detected = True
                    except ValueError:
                        change_detected = False
                model_cont += 1

            if change_detected:
                self.n_drifts_detected += 1
                if self.drift_check == "model":
                    max_error_idx = max(
                        range(len(self.drift_detectors[module])),
                        key=lambda j: self.drift_detectors[module][j].estimation,
                    )
                    models[max_error_idx] = copy.deepcopy(models[max_error_idx])
                    self.cfc[module] = copy.deepcopy(self.cfc[module])
                    self.drift_detectors[module][max_error_idx] = ADWIN()
                    
                elif self.drift_check == "module":
                    module = copy.deepcopy(module)
                    self.drift_detectors[module] = [copy.deepcopy(ADWIN()) for _ in range(model_cont)]
            else:
                x_new = x | predictions     # Create new input by appending predictions to current input
                self.cfc[module].learn_one(x_new, y)
            
        return self


class RensemblerClassifier(BaseRensembler, base.Classifier):
    """Online Rensembler for classification.
    For each incoming observation, each model's `learn_one` method is called `k` times where
    `k` is sampled from a Poisson distribution of parameter equal to "lam".
    Parameters
    ----------
    module_tuples
        Dictionary describing of the shape [(M_1, (base_classifier, N_1)), ...], where M_x is the number of modules of a given type
        and N_x the amount of base_classifier in each of those modules.
    cfc
        Meta-classifier to use in each module.
    lam
        Parameter of the Poisson distribution used during sampling.
    seed
        Random number generator seed for reproducibility.
    drift_check
        "off": Don't check for concept drift.
        "model": If concept drift is detected replace worst base classifier and module meta-classifier for new ones.
        "module": If concept drift is detected replace module for new one.
    unanimity_check
        If greater than 0.0, use given value as threshold to check if there's unanimity in base classifier answer and ignore meta-classifier in such case.

    Examples
    --------
    In the following example three logistic regressions are bagged together. The performance is
    slightly better than when using a single logistic regression.
    >>> from river import datasets
    >>> from river import ensemble
    >>> from river import evaluate
    >>> from river import linear_model
    >>> from river import metrics
    >>> from river import preprocessing
    >>> dataset = datasets.Phishing()
    >>> base_classifier =  preprocessing.StandardScaler() | linear_model.LogisticRegression()
    >>> meta_classifier = preprocessing.StandardScaler() | linear_model.LogisticRegression()    
    >>> models_3x5 = [(3, (base_classifier, 5))]
    >>> model = ensemble.RensemblerClassifier(models_3x5, meta_classifier, lam=1.0, seed=42, unanimity_check=False, drift_check = "off")
    >>> metric = metrics.F1()
    >>> evaluate.progressive_val_score(dataset, model, metric)
    F1: 87.89%
    """

    def is_unanimous(self, predictions: list):
        agg = Counter()
        unanimous = False
        size = len(predictions)
        predIter = iter(predictions)
        while (pred := next(predIter, None)) is not None and not unanimous:
            agg[pred] += 1
            if int(agg.most_common(1)[0][1]) > size*self.unanimity_check:
                unanimous = True

        if unanimous:
            self.n_unanimities_detected += 1
            res = agg.most_common(1)[0][0]
        else:
            res = False
        return res
    
    def predict_proba_one(self, x):
        """Averages the predictions of each classifier."""

        y_pred = collections.Counter()
        for module in self.modules:
            predictions = {}
            unanimity = False
            for model in self.modules[module]:
                predictions[f'prediction_{model}'] = model.predict_one(x)

            if self.unanimity_check:
                unanimity = self.is_unanimous(predictions.values())

            if unanimity:
                module_pred = { unanimity: 1.0 }   #If unanimity give total certainty
                y_pred.update(module_pred)
            else:
                x_new = x | predictions
                cfc_prediction = self.cfc[module].predict_proba_one(x_new)
                y_pred.update(cfc_prediction)

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred