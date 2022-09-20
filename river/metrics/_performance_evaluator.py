from collections import deque
from timeit import default_timer as timer

import numpy as np

from river import metrics


class _ClassificationReport:
    """ "Classification report

    Incrementally tracks classification performance and provide, at any moment, updated
    performance metrics. This performance evaluator is designed for single-output
    (binary and multi-class) classification tasks.

    Parameters
    ----------
    cm: ConfusionMatrix, optional (default=None)
        Confusion matrix instance.

    Examples
    --------
    >>> from river.metrics import _ClassificationReport

    >>> y_true = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    >>> y_pred = [0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2]

    >>> report = _ClassificationReport()

    >>> for i in range(len(y_true)):
    ...     report.add_result(y_true[i], y_pred[i])

    >>> report
    Classification report
    <BLANKLINE>
    n_classes:		         3
    n_samples:		        25
    Accuracy:		    0.4800
    Kappa:			    0.2546
    KappaT:			   -5.5000
    KappaM:			    0.1333
    Precision:		    0.6667
    Recall:			    0.2000
    F1:				    0.3077
    GeometricMean:	    0.4463
    MicroPrecision:	    0.4800
    MicroRecall:	    0.4800
    MicroF1:		    0.4800
    MacroPrecision:	    0.5470
    MacroRecall:	    0.5111
    MacroF1:		    0.4651

    """

    # Define the format specification used for string representation.
    _fmt = ">10.4f"

    def __init__(self, cm: "metrics.ConfusionMatrix" = None):

        self.cm = metrics.ConfusionMatrix() if cm is None else cm
        self.accuracy = metrics.Accuracy(cm=self.cm)
        self.kappa = metrics.CohenKappa(cm=self.cm)
        self.kappa_m = metrics.KappaM(cm=self.cm)
        self.kappa_t = metrics.KappaT(cm=self.cm)
        self.recall = metrics.Recall(cm=self.cm)
        self.micro_recall = metrics.MicroRecall(cm=self.cm)
        self.macro_recall = metrics.MacroRecall(cm=self.cm)
        self.precision = metrics.Precision(cm=self.cm)
        self.micro_precision = metrics.MicroPrecision(cm=self.cm)
        self.macro_precision = metrics.MacroPrecision(cm=self.cm)
        self.f1 = metrics.F1(cm=self.cm)
        self.micro_f1 = metrics.MicroF1(cm=self.cm)
        self.macro_f1 = metrics.MacroF1(cm=self.cm)
        self.geometric_mean = metrics.GeometricMean(cm=self.cm)

    def add_result(self, y_true, y_pred, sample_weight=1.0):
        self.cm.update(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

    def accuracy_score(self):
        return self.accuracy.get()

    def kappa_score(self):
        return self.kappa.get()

    def kappa_t_score(self):
        return self.kappa_t.get()

    def kappa_m_score(self):
        return self.kappa_m.get()

    def precision_score(self):
        return self.precision.get()

    def recall_score(self):
        return self.recall.get()

    def f1_score(self):
        return self.f1.get()

    def geometric_mean_score(self):
        return self.geometric_mean.get()

    def get_last(self):
        return self.cm.last_y_true, self.cm.last_y_pred

    @property
    def n_samples(self):
        return self.cm.n_samples

    @property
    def n_classes(self):
        return self.cm.n_classes

    @property
    def confusion_matrix(self):
        return self.cm

    def reset(self):
        self.cm.reset()

    def __repr__(self):
        return "".join(
            [
                "Classification report\n\n",
                f"n_classes:\t\t{self.n_classes:>10}\n",
                f"n_samples:\t\t{self.n_samples:>10}\n",
                "\n",
                self._info(),
            ]
        )

    def _info(self):
        return "".join(
            [
                f"Accuracy:\t\t{self.accuracy.get():{self._fmt}}\n",
                f"Kappa:\t\t\t{self.kappa.get():{self._fmt}}\n",
                f"KappaT:\t\t\t{self.kappa_t.get():{self._fmt}}\n",
                f"KappaM:\t\t\t{self.kappa_m.get():{self._fmt}}\n",
                f"Precision:\t\t{self.precision.get():{self._fmt}}\n",
                f"Recall:\t\t\t{self.recall.get():{self._fmt}}\n",
                f"F1:\t\t\t\t{self.f1.get():{self._fmt}}\n",
                f"GeometricMean:\t{self.geometric_mean.get():{self._fmt}}\n",
                "\n",
                f"MicroPrecision:\t{self.micro_precision.get():{self._fmt}}\n",
                f"MicroRecall:\t{self.micro_recall.get():{self._fmt}}\n",
                f"MicroF1:\t\t{self.micro_f1.get():{self._fmt}}\n",
                f"MacroPrecision:\t{self.macro_precision.get():{self._fmt}}\n",
                f"MacroRecall:\t{self.macro_recall.get():{self._fmt}}\n",
                f"MacroF1:\t\t{self.macro_f1.get():{self._fmt}}\n",
            ]
        )


class _RollingClassificationReport(_ClassificationReport):
    """Rolling classification report

    Incrementally tracks classification performance over a sliding window and provide,
    at any moment, updated performance metrics. This performance evaluator is designed
    for single-output (binary and multi-class) classification tasks.

    Parameters
    ----------
    cm: ConfusionMatrix, optional (default=None)
        Confusion matrix instance.

    window_size: int
        Window size.

    Examples
    --------
    >>> from river.metrics import _RollingClassificationReport

    >>> y_true = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    >>> y_pred = [0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2]

    >>> report = _RollingClassificationReport(window_size=20)

    >>> for i in range(len(y_true)):
    ...     report.add_result(y_true[i], y_pred[i])

    >>> report
    Rolling classification report
    <BLANKLINE>
    n_classes:		         3
    n_samples:		        20
    window_size:	        20
    Accuracy:		    0.4000
    Kappa:			    0.1696
    KappaT:			   -5.0000
    KappaM:			    0.2000
    Precision:		    0.6667
    Recall:			    0.2000
    F1:				    0.3077
    GeometricMean:	    0.0000
    MicroPrecision:	    0.4000
    MicroRecall:	    0.4000
    MicroF1:		    0.4000
    MacroPrecision:	    0.4722
    MacroRecall:	    0.2889
    MacroF1:		    0.3379

    """

    def __init__(self, cm: "metrics.ConfusionMatrix" = None, window_size=200):
        self.window_size = window_size
        self._rolling_cm = metrics.Rolling(
            metrics.ConfusionMatrix() if cm is None else cm,
            window_size=self.window_size,
        )
        super().__init__(cm=self._rolling_cm.metric)

    def add_result(self, y_true, y_pred, sample_weight=1.0):
        self._rolling_cm.update(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
        )

    def __repr__(self):
        return "".join(
            [
                "Rolling classification report\n\n",
                f"n_classes:\t\t{self.n_classes:>10}\n",
                f"n_samples:\t\t{self.n_samples:>10}\n",
                f"window_size:\t{self.window_size:>10}\n",
                "\n",
                self._info(),
            ]
        )


class _MLClassificationReport:
    """Multi-label classification report.

    Incrementally tracks a classifier's performance and provide, at any moment, updated
    performance metrics. This performance evaluator is designed for multi-output
    (multi-label) classification tasks.

    Parameters
    ----------
    cm: MultiLabelConfusionMatrix, optional (default=None)
        Multi-label confusion matrix instance.

    Examples
    --------
    >>> from river.metrics import _MLClassificationReport

    >>> y_0 = [True]*100
    >>> y_1 = [True]*90 + [False]*10
    >>> y_2 = [True]*85 + [False]*10 + [True]*5
    >>> y_true = []
    >>> y_pred = []
    >>> for i in range(len(y_0)):
    ...     y_true.append({0:True, 1:True, 2:True})
    ...     y_pred.append({0:y_0[i], 1:y_1[i], 2:y_2[i]})

    >>> evaluator = _MLClassificationReport()

    >>> for i in range(len(y_true)):
    ...     evaluator.add_result(y_true[i], y_pred[i])

    >>> evaluator
    Multi-label classification report
    <BLANKLINE>
    n_classes:		         3
    n_samples:		       100
    Hamming:		    0.9333
    HammingLoss:	    0.0667
    ExactMatch:		    0.8500
    JaccardIndex:	    0.9333

    """

    # Define the format specification used for string representation.
    _fmt = ">10.4f"

    def __init__(self, cm: "metrics.MultiLabelConfusionMatrix" = None):
        self.cm = metrics.MultiLabelConfusionMatrix() if cm is None else cm
        self.hamming = metrics.Hamming(cm=self.cm)
        self.hamming_loss = metrics.HammingLoss(cm=self.cm)
        self.jaccard_index = metrics.Jaccard(cm=self.cm)
        self.exact_match = metrics.ExactMatch(cm=self.cm)

    def add_result(self, y_true, y_pred, sample_weight=1.0):
        self.cm.update(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

    def hamming_score(self):
        return self.hamming.get()

    def hamming_loss_score(self):
        return self.hamming_loss.get()

    def exact_match_score(self):
        return self.exact_match.get()

    def jaccard_score(self):
        return self.jaccard_index.get()

    def get_last(self):
        return self.cm.last_y_true, self.cm.last_y_pred

    @property
    def n_samples(self):
        return self.cm.n_samples

    @property
    def n_labels(self):
        return self.cm.n_labels

    @property
    def confusion_matrix(self):
        return self.cm

    def reset(self):
        self.cm.reset()

    def __repr__(self):
        return "".join(
            [
                "Multi-label classification report\n\n",
                f"n_classes:\t\t{self.n_labels:>10}\n",
                f"n_samples:\t\t{self.n_samples:>10}\n",
                "\n",
                self._info(),
            ]
        )

    def _info(self):
        return "".join(
            [
                f"Hamming:\t\t{self.hamming.get():{self._fmt}}\n",
                f"HammingLoss:\t{self.hamming_loss.get():{self._fmt}}\n",
                f"ExactMatch:\t\t{self.exact_match.get():{self._fmt}}\n",
                f"JaccardIndex:\t{self.jaccard_index.get():{self._fmt}}\n",
            ]
        )


class _RollingMLClassificationReport(_MLClassificationReport):
    """Rolling multi-label classification report.

    Incrementally tracks a classifier's performance over a sliding window and provide,
    at any moment, updated performance metrics. This performance evaluator is designed
    for multi-output (multi-label) classification tasks.

    Parameters
    ----------
    cm: MultiLabelConfusionMatrix, optional (default=None)
        Multi-label confusion matrix instance.

    window_size: int
        Window size.

    Examples
    --------
    >>> from river.metrics import _RollingMLClassificationReport

    >>> y_0 = [True]*100
    >>> y_1 = [True]*90 + [False]*10
    >>> y_2 = [True]*85 + [False]*10 + [True]*5
    >>> y_true = []
    >>> y_pred = []
    >>> for i in range(len(y_0)):
    ...     y_true.append({0:True, 1:True, 2:True})
    ...     y_pred.append({0:y_0[i], 1:y_1[i], 2:y_2[i]})

    >>> evaluator = _RollingMLClassificationReport(window_size=20)
    >>> for i in range(len(y_true)):
    ...     evaluator.add_result(y_true[i], y_pred[i])

    >>> evaluator
    Rolling multi-label classification report
    <BLANKLINE>
    n_labels:		         3
    n_samples:		        20
    window_size:	        20
    Hamming:		    0.6667
    HammingLoss:	    0.3333
    ExactMatch:		    0.2500
    JaccardIndex:	    0.6667

    """

    def __init__(self, cm: "metrics.ConfusionMatrix" = None, window_size=200):
        self.window_size = window_size
        self._rolling_cm = metrics.Rolling(
            metrics.MultiLabelConfusionMatrix() if cm is None else cm,
            window_size=self.window_size,
        )
        super().__init__(cm=self._rolling_cm.metric)

    def add_result(self, y_true, y_pred, sample_weight=1.0):
        self._rolling_cm.update(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
        )

    def __repr__(self):
        return "".join(
            [
                "Rolling multi-label classification report\n\n",
                f"n_labels:\t\t{self.n_labels:>10}\n",
                f"n_samples:\t\t{self.n_samples:>10}\n",
                f"window_size:\t{self.window_size:>10}\n",
                "\n",
                self._info(),
            ]
        )


class _ClusteringReport:
    """ "Clustering report

    Incrementally tracks clustering performance and provide, at any moment, updated
    performance metrics. Note that clustering algorithms only concern single-output tasks.

    Parameters
    ----------
    cm: ConfusionMatrix, optional (default=None)
        Confusion matrix instance.

    Examples
    --------
    >>> from river.metrics import _ClusteringReport

    >>> y_true = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    >>> y_pred = [0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2]

    >>> report = _ClusteringReport()

    >>> for i in range(len(y_true)):
    ...     report.add_result(y_true[i], y_pred[i])

    >>> report
    Clustering report
    <BLANKLINE>
    n_classes:                       3
    n_clusters:                      3
    n_samples:                      25
    PairConfusionMatrix:
    {0: defaultdict(<class 'int'>, {0: 256.0, 1: 152.0}), 1: defaultdict(<class 'int'>, {0: 110.0, 1: 82.0})}
    <BLANKLINE>
    MatthewsCorrCoef:           0.2872
    Completeness:               0.1505
    Homogeneity:                0.1345
    VBeta:                      0.1420
    Q0:                         1.7832
    Q2:                         0.2586
    PrevalenceThreshold:        0.4027
    MutualInfo:                 0.1448
    NormalizedMutualInfo:       0.1420
    AdjustedMutualInfo:         0.0509
    Rand:                       0.5633
    AdjustedRand:               0.0515
    VariationInfo:              2.5240
    <BLANKLINE>

    """

    # Define the format specification used for string representation.
    _fmt = ">10.4f"

    def __init__(self, cm: "metrics.ConfusionMatrix" = None):

        self.cm = metrics.ConfusionMatrix() if cm is None else cm
        self.pair_cm = metrics.PairConfusionMatrix(self.cm)

        self.matthews_corr = metrics.MatthewsCorrCoef(cm=self.cm)
        self.completeness = metrics.Completeness(cm=self.cm)
        self.homogeneity = metrics.Homogeneity(cm=self.cm)
        self.vbeta = metrics.VBeta(cm=self.cm)
        self.q0 = metrics.Q0(cm=self.cm)
        self.q2 = metrics.Q2(cm=self.cm)
        self.pt = metrics.PrevalenceThreshold(cm=self.cm)
        self.mutual_info = metrics.MutualInfo(cm=self.cm)
        self.normalized_mutual_info = metrics.NormalizedMutualInfo(cm=self.cm)
        self.adjusted_mutual_info = metrics.AdjustedMutualInfo(cm=self.cm)
        self.rand = metrics.Rand(cm=self.cm)
        self.adjusted_rand = metrics.AdjustedRand(cm=self.cm)
        self.variation_info = metrics.VariationInfo(cm=self.cm)

    def add_result(self, y_true, y_pred, sample_weight=1.0):
        self.cm.update(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)

    def matthews_corr_coef(self):
        return self.matthews_corr.get()

    def completeness_score(self):
        return self.completeness.get()

    def homogeneity_score(self):
        return self.homogeneity.get()

    def vbeta_score(self):
        return self.vbeta.get()

    def q0_score(self):
        return self.q0.get()

    def q2_score(self):
        return self.q2.get()

    def prevalence_threshold(self):
        return self.pt.get()

    def mutual_info_score(self):
        return self.mutual_info.get()

    def normalized_mutual_info_score(self):
        return self.normalized_mutual_info.get()

    def adjusted_mutual_info_score(self):
        return self.adjusted_mutual_info.get()

    def rand_score(self):
        return self.rand.get()

    def adjusted_rand_score(self):
        return self.adjusted_rand.get()

    def variation_info_score(self):
        return self.variation_info.get()

    def get_last(self):
        return self.cm.last_y_true, self.cm.last_y_pred

    @property
    def n_samples(self):
        return self.cm.n_samples

    @property
    def n_classes(self):
        return len([i for i in self.cm.sum_row.values() if i != 0])

    @property
    def n_clusters(self):
        return len([i for i in self.cm.sum_col.values() if i != 0])

    @property
    def confusion_matrix(self):
        return self.cm

    def pair_confusion_matrix(self):
        return self.pair_cm.get()

    def reset(self):
        self.cm.reset()

    def __repr__(self):
        return "".join(
            [
                "Clustering report\n\n",
                f"n_classes:\t\t\t\t{self.n_classes:>10}\n",
                f"n_clusters:\t\t\t\t{self.n_clusters:>10}\n",
                f"n_samples:\t\t\t\t{self.n_samples:>10}\n",
                "PairConfusionMatrix:",
                "\n",
                f"{self.pair_cm.get()}",
                "\n",
                "\n",
                self._info(),
            ]
        )

    def _info(self):
        return "".join(
            [
                f"MatthewsCorrCoef:\t\t{self.matthews_corr.get():{self._fmt}}\n",
                f"Completeness:\t\t\t{self.completeness.get():{self._fmt}}\n",
                f"Homogeneity:\t\t\t{self.homogeneity.get():{self._fmt}}\n",
                f"VBeta:\t\t\t\t\t{self.vbeta.get():{self._fmt}}\n",
                f"Q0:\t\t\t\t\t\t{self.q0.get():{self._fmt}}\n",
                f"Q2:\t\t\t\t\t\t{self.q2.get():{self._fmt}}\n",
                f"PrevalenceThreshold:\t{self.pt.get():{self._fmt}}\n",
                f"MutualInfo:\t\t\t\t{self.mutual_info.get():{self._fmt}}\n",
                f"NormalizedMutualInfo:\t{self.normalized_mutual_info.get():{self._fmt}}\n",
                f"AdjustedMutualInfo:\t\t{self.adjusted_mutual_info.get():{self._fmt}}\n",
                f"Rand:\t\t\t\t\t{self.rand.get():{self._fmt}}\n",
                f"AdjustedRand:\t\t\t{self.adjusted_rand.get():{self._fmt}}\n",
                f"VariationInfo:\t\t\t{self.variation_info.get():{self._fmt}}\n",
            ]
        )


class _RollingClusteringReport(_ClusteringReport):
    """Rolling clustering report

    Incrementally tracks clustering performance over a sliding window and provide,
    at any moment, updated performance metrics. Note that, clustering algorithms
    only concern single-output tasks.

    Parameters
    ----------
    cm: ConfusionMatrix, optional (default=None)
        Confusion matrix instance.

    window_size: int
        Window size.

    Examples
    --------
    >>> from river.metrics import _RollingClusteringReport

    >>> y_true = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    >>> y_pred = [0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2]

    >>> report = _RollingClusteringReport(window_size=20)

    >>> for i in range(len(y_true)):
    ...     report.add_result(y_true[i], y_pred[i])

    >>> report
    Rolling Clustering report
    <BLANKLINE>
    n_classes:                       3
    n_clusters:                      3
    n_samples:                      20
    window_size:                    20
    PairConfusionMatrix:
    {0: defaultdict(<class 'int'>, {0: 154.0, 1: 64.0}), 1: defaultdict(<class 'int'>, {0: 92.0, 1: 70.0})}
    <BLANKLINE>
    MatthewsCorrCoef:           0.2116
    Completeness:               0.2463
    Homogeneity:                0.2908
    VBeta:                      0.2667
    Q0:                         1.3813
    Q2:                         0.3365
    PrevalenceThreshold:        0.4580
    MutualInfo:                 0.2488
    NormalizedMutualInfo:       0.2667
    AdjustedMutualInfo:         0.1646
    Rand:                       0.5895
    AdjustedRand:               0.1417
    VariationInfo:              1.9742
    <BLANKLINE>

    """

    def __init__(self, cm: "metrics.ConfusionMatrix" = None, window_size=200):
        self.window_size = window_size
        self._rolling_cm = metrics.Rolling(
            metrics.ConfusionMatrix() if cm is None else cm,
            window_size=self.window_size,
        )
        super().__init__(cm=self._rolling_cm.metric)

    def add_result(self, y_true, y_pred, sample_weight=1.0):
        self._rolling_cm.update(
            y_true=y_true, y_pred=y_pred, sample_weight=sample_weight
        )

    def __repr__(self):
        return "".join(
            [
                "Rolling Clustering report\n\n",
                f"n_classes:\t\t\t\t{self.n_classes:>10}\n",
                f"n_clusters:\t\t\t\t{self.n_clusters:>10}\n",
                f"n_samples:\t\t\t\t{self.n_samples:>10}\n",
                f"window_size:\t\t\t{self.window_size:>10}\n",
                "PairConfusionMatrix:\n",
                f"{self.pair_cm.get()}\n",
                "\n",
                self._info(),
            ]
        )


class _RegressionReport(object):
    """Regression report.

     Keeps incremental performance metrics in a regression problem.

    Examples
    --------
    >>> import numpy
    >>> from river.metrics import _RegressionReport
    >>>
    >>> y_true = numpy.sin(range(100))
    >>> y_pred = numpy.sin(range(100)) + .05
    >>>
    >>> measurements = _RegressionReport()
    >>>
    >>> for y_t, y_p in zip(y_true, y_pred):
    ...     measurements.add_result(y_t, y_p)
    >>>
    >>> measurements
    Regression report
    <BLANKLINE>
    n_samples:		       100
    MAE:				0.050000
    MSE:				0.002500
     R2:                0.995001

    """

    # Define the format specification used for string representation.
    _fmt = ",.6f"  # Use commas to separate big numbers and show 6 decimals

    def __init__(self):
        super().__init__()
        self.mae = metrics.MAE()
        self.mse = metrics.MSE()
        self.r2 = metrics.R2()
        self.last_true_label = None
        self.last_prediction = None

    def reset(self):
        self.mae = metrics.MAE()
        self.mse = metrics.MSE()
        self.r2 = metrics.R2()
        self.last_true_label = None
        self.last_prediction = None

    def add_result(self, y_true, y_pred):
        self.last_true_label = y_true
        self.last_prediction = y_pred

        self.mae.update(y_true, y_pred)
        self.mse.update(y_true, y_pred)
        self.r2.update(y_true, y_pred)

    def get_average_error(self):
        return self.mae.get()

    def get_mean_square_error(self):
        return self.mse.get()

    def get_r2_score(self):
        return self.r2.get()

    def get_last(self):
        return self.last_true_label, self.last_prediction

    @property
    def n_samples(self):
        return int(self.mae._mean.n)

    def __repr__(self):
        return "".join(
            [
                "Regression report\n\n",
                f"n_samples:\t\t{self.n_samples:>10}\n",
                "\n",
                self._info(),
            ]
        )

    def _info(self):
        return "".join(
            [
                f"MAE:\t\t\t\t{self.mae.get():{self._fmt}}\n",
                f"MSE:\t\t\t\t{self.mse.get():{self._fmt}}\n",
                f" R2:\t\t\t\t{self.r2.get():{self._fmt}}\n",
            ]
        )


class _RollingRegressionReport(_RegressionReport):
    """Rolling regression report

    Keeps incremental performance metrics in a regression problem over a fixed-size window.

    Examples
    --------
    >>> import numpy
    >>> from river.metrics import _RollingRegressionReport
    >>>
    >>> y_true = numpy.sin(range(100))
    >>> y_pred = numpy.sin(range(100)) + .05
    >>>
    >>> measurements = _RollingRegressionReport(window_size=20)
    >>>
    >>> for y_t, y_p in zip(y_true, y_pred):
    ...     measurements.add_result(y_t, y_p)
    >>>
    >>> measurements
    Rolling regression report
    <BLANKLINE>
    n_samples:		        20
    window_size:            20
    MAE:				0.050000
    MSE:				0.002500
     R2:                0.995228

    """

    def __init__(self, window_size=200):
        super().__init__()
        self.window_size = window_size
        self.mae = metrics.Rolling(metrics.MAE(), window_size=self.window_size)
        self.mse = metrics.Rolling(metrics.MSE(), window_size=self.window_size)
        self.r2 = metrics.Rolling(metrics.R2(), window_size=self.window_size)
        self.sample_count = 0
        self.last_true_label = None
        self.last_prediction = None

    def reset(self):
        self.mae = metrics.Rolling(metrics.MAE(), window_size=self.window_size)
        self.mse = metrics.Rolling(metrics.MSE(), window_size=self.window_size)
        self.r2 = metrics.Rolling(metrics.R2(), window_size=self.window_size)
        self.sample_count = 0
        self.last_true_label = None
        self.last_prediction = None

    @property
    def n_samples(self):
        return int(self.mae._metric._mean.n)

    def __repr__(self):
        return "".join(
            [
                "Rolling regression report\n\n",
                f"n_samples:\t\t{self.n_samples:>10}\n",
                f"window_size:\t{self.window_size:>10}\n",
                "\n",
                self._info(),
            ]
        )


class _MTRegressionReport:
    """Multi-target regression report

    Keeps incremental performance metrics in a multi-target regression problem.

    Examples
    --------
    >>> import numpy
    >>> from river.metrics import _MTRegressionReport
    >>>
    >>> y_true = numpy.zeros((100, 3))
    >>> y_pred = numpy.zeros((100, 3))
    >>> for t in range(3):
    ...     y_true[:, t] = numpy.sin(range(100))
    ...     y_pred[:, t] = numpy.sin(range(100)) + (t + 1) * .05
    >>>
    >>> measurements = _MTRegressionReport()
    >>>
    >>> for y_t, y_p in zip(y_true, y_pred):
    ...     measurements.add_result(y_t, y_p)
    >>>
    >>> measurements
    Multi-target regression report
    <BLANKLINE>
    n_targets:		         3
    n_samples:		       100
    Average MAE:			0.100000
    Average MSE:			0.011667
    Average RMSE:			0.100000

    """

    # Define the format specification used for string representation.
    _fmt = ",.6f"  # Use commas to separate big numbers and show 6 decimals

    def __init__(self):
        super().__init__()
        self.n_targets = 0
        self.total_square_error = 0.0
        self.average_error = 0.0
        self.n_samples = 0
        self.last_true_label = None
        self.last_prediction = None

    def reset(self):
        self.n_targets = 0
        self.total_square_error = 0.0
        self.average_error = 0.0
        self.n_samples = 0
        self.last_true_label = None
        self.last_prediction = None

    def add_result(self, y_true, y_pred):
        self.last_true_label = y_true
        self.last_prediction = y_pred

        self.n_targets = y_true.size

        self.total_square_error += (y_true - y_pred) * (y_true - y_pred)
        self.average_error += np.absolute(y_true - y_pred)
        self.n_samples += 1

    def get_average_mean_square_error(self):
        try:
            return np.sum(self.total_square_error / self.n_samples) / self.n_targets
        except ZeroDivisionError:
            return 0.0

    def get_average_absolute_error(self):
        try:
            return np.sum(self.average_error / self.n_samples) / self.n_targets
        except ZeroDivisionError:
            return 0.0

    def get_average_root_mean_square_error(self):
        try:
            mse = self.total_square_error / self.n_samples
            return (
                np.sum(np.sqrt(mse, out=np.zeros_like(mse), where=mse >= 0.0))
                / self.n_targets
            )
        except ZeroDivisionError:
            return 0.0

    def get_last(self):
        return self.last_true_label, self.last_prediction

    def __repr__(self):
        return "".join(
            [
                "Multi-target regression report\n\n",
                f"n_targets:\t\t{self.n_targets:>10}\n",
                f"n_samples:\t\t{self.n_samples:>10}\n",
                "\n",
                self._info(),
            ]
        )

    def _info(self):
        return "".join(
            [
                f"Average MAE:\t\t\t{self.get_average_absolute_error():{self._fmt}}\n",
                f"Average MSE:\t\t\t{self.get_average_mean_square_error():{self._fmt}}\n",
                f"Average RMSE:\t\t\t{self.get_average_root_mean_square_error():{self._fmt}}\n",
            ]
        )


class _RollingMTRegressionReport(_MTRegressionReport):
    """Rolling multi-target regression report

    Keeps incremental performance metrics in a multi-target regression problem over a
    fixed-size window.

    Examples
    --------
    >>> import numpy
    >>> from river.metrics import _RollingMTRegressionReport
    >>>
    >>> y_true = numpy.zeros((100, 3))
    >>> y_pred = numpy.zeros((100, 3))
    >>> for t in range(3):
    ...     y_true[:, t] = numpy.sin(range(100))
    ...     y_pred[:, t] = numpy.sin(range(100)) + (t + 1) * .05
    >>>
    >>> measurements = _RollingMTRegressionReport(window_size=20)
    >>>
    >>> for y_t, y_p in zip(y_true, y_pred):
    ...     measurements.add_result(y_t, y_p)
    >>>
    >>> measurements
    Rolling multi-target regression report
    <BLANKLINE>
    n_targets:		         3
    n_samples:		        20
    window_size:	        20
    Average MAE:			0.100000
    Average MSE:			0.011667
    Average RMSE:			0.100000

    """

    def __init__(self, window_size=200):
        super().__init__()
        self.n_targets = 0
        self.total_square_error = 0.0
        self.average_error = 0.0
        self.last_true_label = None
        self.last_prediction = None
        self.window_size = window_size
        self.total_square_error_correction = deque(maxlen=self.window_size)
        self.average_error_correction = deque(maxlen=self.window_size)

    def reset(self):
        self.total_square_error = 0.0
        self.average_error = 0.0
        self.last_true_label = None
        self.last_prediction = None
        self.total_square_error_correction = deque(maxlen=self.window_size)
        self.average_error_correction = deque(maxlen=self.window_size)

    def add_result(self, y_true, y_pred):
        self.last_true_label = y_true
        self.last_prediction = y_pred

        self.n_targets = len(y_true)

        self.total_square_error += (y_true - y_pred) * (y_true - y_pred)
        self.average_error += np.absolute(y_true - y_pred)

        old_square = (
            self.total_square_error_correction[0]
            if len(self.total_square_error_correction) == self.window_size
            else None
        )
        self.total_square_error_correction.append(
            -1 * (y_true - y_pred) * (y_true - y_pred)
        )
        old_average = (
            self.average_error_correction[0]
            if len(self.average_error_correction) == self.window_size
            else None
        )
        self.average_error_correction.append(-1 * (np.absolute(y_true - y_pred)))

        if (old_square is not None) and (old_average is not None):
            self.total_square_error += old_square
            self.average_error += old_average

    @property
    def n_samples(self):
        return len(self.total_square_error_correction)

    @n_samples.setter
    def n_samples(self, value):
        pass

    def __repr__(self):
        return "".join(
            [
                "Rolling multi-target regression report\n\n",
                f"n_targets:\t\t{self.n_targets:>10}\n",
                f"n_samples:\t\t{self.n_samples:>10}\n",
                f"window_size:\t{self.window_size:>10}\n",
                "\n",
                self._info(),
            ]
        )


class RunningTimeMeasurements(object):
    """Class used to compute the running time of a predictive model.

    The training, prediction, and total time are considered separately. The
    class accounts for the amount of time each model effectively spent on
    training and testing. To do so, timers for each of the actions are
    considered.

    Besides the properties getters, the available compute time methods
    must be used as follows:

    - `compute_{training, testing}_time_begin`
    - Perform training/action
    - `compute_{training, testing}_time_end`

    Additionally, the `update_time_measurements` method updates the total
    running time accounting, as well as, the total seen samples count.
    """

    def __init__(self):
        super().__init__()
        self._training_start = None
        self._testing_start = None
        self._training_time = 0
        self._testing_time = 0
        self._sample_count = 0
        self._total_time = 0

    def reset(self):
        self._training_time = 0
        self._testing_time = 0
        self._sample_count = 0
        self._total_time = 0

    def compute_training_time_begin(self):
        """Start measuring training time."""
        self._training_start = timer()

    def compute_training_time_end(self):
        """Finish measuring training time."""
        self._training_time += timer() - self._training_start

    def compute_testing_time_begin(self):
        """Start measuring testing time."""
        self._testing_start = timer()

    def compute_testing_time_end(self):
        """Finish measuring testing time."""
        self._testing_time += timer() - self._testing_start

    def update_time_measurements(self, increment=1):
        """Update the current total running time."""
        if increment > 0:
            self._sample_count += increment
        else:
            self._sample_count += 1

        self._total_time = self._training_time + self._testing_time

    def get_current_training_time(self):
        return self._training_time

    def get_current_testing_time(self):
        return self._testing_time

    def get_current_total_running_time(self):
        return self._total_time

    def get_info(self):
        return (
            "RunningTimeMeasurements: sample_count: "
            + str(self._sample_count)
            + " - Total running time: "
            + str(self.get_current_total_running_time())
            + " - training_time: "
            + str(self.get_current_training_time())
            + " - testing_time: "
            + str(self.get_current_testing_time())
        )
