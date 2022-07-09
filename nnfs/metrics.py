#! /usr/bin/env python3.10

"""
Implementation of various metrics.
`numpy` is required. -> https://numpy.org/
Compatible with python3.10+.
Mahyar@Mahyar24.com, Sat 23 Apr 2022.
"""
import abc

import numpy as np


class Metric(metaclass=abc.ABCMeta):
    """
    Abstract class for metrics.
    """

    @staticmethod
    def check_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        assert len(y_true.shape) == 1, f"y ({y_true.shape=}) must be a 1D array."
        assert len(y_pred.shape) == 1, f"y_pred ({y_pred.shape=}) must be a 1D array."
        assert (
            y_true.shape[0] == y_pred.shape[0]
        ), f"y_true ({y_true.shape=}) and y_pred ({y_pred.shape=}) must have the same number of samples."

    @staticmethod
    @abc.abstractmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the metric.
        """
        pass


class Accuracy(Metric):
    """
    Accuracy metric for classification problems.
    """

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the accuracy of classification.
        """
        Metric.check_shapes(y_true, y_pred)

        return (y_pred == y_true).mean()


class Precision(Metric):
    """
    Precision metric for classification problems.
    Precision: `TP / (TP + FP)`
    """

    @staticmethod
    def _evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Precision of binary classification.
        Precision: `TP / (TP + FP)` where `TP`: True Positives, `FP`: False Positives.
        """
        tp = np.where(y_true, y_pred, 0).sum()
        fp = np.where(y_pred, np.where(y_true, 0, 1), 0).sum()

        # In case of zero division, we will return 0.
        if tp + fp == 0:
            return 0.0

        return tp / (tp + fp)

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Precision of classification.
        In case of multilabel targets, we will return macro average of precisions.
        Precision: `TP / (TP + FP)` where `TP`: True Positives, `FP`: False Positives.
        """
        Metric.check_shapes(y_true, y_pred)

        labels = set(y_true) | set(y_pred)

        scores = np.empty((len(labels),))

        for i, label in enumerate(labels):
            y_true_mod = np.where(y_true != label, 0, 1)
            y_pred_mod = np.where(y_pred != label, 0, 1)
            scores[i] = Precision._evaluate_binary(y_true_mod, y_pred_mod)

        return scores.mean()


class Recall(Metric):
    """
    Recall metric for classification problems.
    Recall: `TP / (TP + FN)`
    """

    @staticmethod
    def _evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Recall of binary classification.
        Recall: `TP / (TP + FN)` where `TP`: True Positives, `FN`: False Negatives.
        """
        tp = np.where(y_true, y_pred, 0).sum()
        fn = np.where(np.where(y_pred, 0, 1), y_true, 0).sum()

        # In case of zero division, we will return 0.
        if tp + fn == 0:
            return 0.0

        return tp / (tp + fn)

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Recall of classification.
        In case of multilabel targets, we will return macro average of recalls.
        Recall: `TP / (TP + FN)` where `TP`: True Positives, `FN`: False Negatives.
        """
        Metric.check_shapes(y_true, y_pred)

        labels = set(y_true) | set(y_pred)

        scores = np.empty((len(labels),))

        for i, label in enumerate(labels):
            y_true_mod = np.where(y_true != label, 0, 1)
            y_pred_mod = np.where(y_pred != label, 0, 1)
            scores[i] = Recall._evaluate_binary(y_true_mod, y_pred_mod)

        return scores.mean()


class FScore(Metric):
    """
    F-Score metric for classification problems.
    Harmonic mean of precision and recall.
    """

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the F score of classification.
        In case of multilabel targets, we will return macro average of precisions and recalls.
        F-Score: `2 * ((precision * recall) / (precision + recall))`
        """
        precision = Precision.evaluate(y_true, y_pred)
        recall = Recall.evaluate(y_true, y_pred)

        if precision + recall == 0:
            return 0.0

        return 2 * ((precision * recall) / (precision + recall))


class ExplainedVariance(Metric):
    """
    Explained variance metric for regression problems.
    """

    @staticmethod
    def evaluate(y_true, y_pred: np.ndarray) -> float:
        """
        Explained-Variance: `1 - (var(y - y_pred) / var(y))` where `var` is variance.
        """
        Metric.check_shapes(y_true, y_pred)

        return 1 - (np.var(y_true - y_pred) / np.var(y_true))
