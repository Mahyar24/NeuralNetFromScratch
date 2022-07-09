#! /usr/bin/env python3.10

"""
All necessary Loss metrics and implementations.
`numpy` is required. -> https://numpy.org/
Compatible with python3.10+.
Mahyar@Mahyar24.com, Sat 23 Apr 2022.
"""

import abc
import enum

import numpy as np

from nnfs.activations import Softmax
from nnfs.layer import Layer


def check_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Check if shapes of y_true and y_pred are appropriate.
    """
    assert len(y_true.shape) == 1, f"y_true ({y_true.shape=}) must be a 1D array."
    assert len(y_pred.shape) == 2, f"y_pred ({y_pred.shape=}) must be a 2D array."
    assert (
        y_true.shape[0] == y_pred.shape[0]
    ), f"y_true ({y_true.shape=}) and y_pred ({y_pred.shape=}) must have the same number of samples."


@enum.unique
class LossType(enum.Enum):
    """
    Types of loss functions.
    """

    REGRESSION = "REGRESSION"
    CLASSIFICATION = "CLASSIFICATION"


class Loss(metaclass=abc.ABCMeta):
    """
    Abstract base class for all Loss implementations.
    """

    def __int__(self) -> None:
        self.derivatives = None

    @abc.abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the loss value.
        """
        check_shapes(y_true, y_pred)

    @abc.abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Backward pass of the loss function.
        """
        check_shapes(y_true, y_pred)

    @staticmethod
    def regularization(layer: Layer) -> float:
        """
        Calculate the regularization loss for a layer.
        """
        regularization_loss = 0.0

        # L1
        if layer.w_l1 > 0.0:
            regularization_loss += np.abs(layer.w_l1 * layer.weights).sum()
        if layer.b_l1 > 0.0:
            regularization_loss += np.abs(layer.b_l1 * layer.biases).sum()
        # L2
        if layer.w_l2 > 0.0:
            regularization_loss += layer.w_l2 * (layer.weights**2).sum()
        if layer.b_l2 > 0.0:
            regularization_loss += layer.b_l2 * (layer.biases**2).sum()

        return regularization_loss


class CategoricalLoss(Loss):
    """
    Categorical cross-entropy loss.
    """

    loss_type = LossType.CLASSIFICATION

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the loss value.
        loss: `-log(correct_prediction) / batch_size`
        """
        check_shapes(y_true, y_pred)

        # Clip the predictions to avoid NaN.
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        correct_confidences = y_pred_clipped[range(y_true.shape[0]), y_true]
        loss = (-np.log(correct_confidences)).mean()
        return loss

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Backward pass of the loss function.
        Backward: `-(y_true / y_pred)`
        """
        check_shapes(y_true, y_pred)

        one_hot_y_true = np.eye(y_pred.shape[1])[y_true]
        self.derivatives = -one_hot_y_true / y_pred
        # Normalizing the derivatives.
        self.derivatives /= y_true.shape[0]


class BinaryLoss(Loss):
    """
    Binary cross-entropy loss.
    """

    loss_type = LossType.CLASSIFICATION

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the loss value.
        loss: `-y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred) / batch_size`
        """
        check_shapes(y_true, y_pred)

        # Clip the predictions to avoid NaN.
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Reshaping y_true to match y_pred.
        y_true = y_true.copy().reshape(-1, 1)

        losses = -(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        return losses.mean(axis=-1).mean()

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Backward pass of the loss function.
        Backward: `-((y_true / y_pred) - (1 - y_true) / (1 - y_pred)) / number_of_classes`
        """
        check_shapes(y_true, y_pred)

        # Clip the predictions to avoid NaN.
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Reshaping y_true to match y_pred.
        y_true = y_true.copy().reshape(-1, 1)

        self.derivatives = (
            -((y_true / y_pred_clipped) - ((1 - y_true) / (1 - y_pred_clipped)))
            / y_pred_clipped.shape[1]
        )
        # Normalizing the derivatives.
        self.derivatives /= y_true.shape[0]


class MSELoss(Loss):
    """
    Mean squared error loss.
    """

    loss_type = LossType.REGRESSION

    def __init__(self) -> None:
        super().__init__()

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the loss value.
        loss: `(y_true - y_pred) ** 2 / batch_size`
        """
        check_shapes(y_true, y_pred)

        # Reshaping y_true to match y_pred.
        y_true = y_true.copy().reshape(-1, 1)

        losses = (y_true - y_pred) ** 2
        return losses.mean()

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Backward pass of the loss function.
        Backward: `-2 * (y_true - y_pred)`
        """
        check_shapes(y_true, y_pred)

        # Reshaping y_true to match y_pred.
        y_true = y_true.copy().reshape(-1, 1)

        self.derivatives = -2 * (y_true - y_pred)
        # Normalizing the derivatives.
        self.derivatives /= y_true.shape[0]


class SoftmaxLoss:
    """
    Softmax activation layer combined with categorical cross-entropy loss.
    The derivatives of combining these two layers is computationally a lot cheaper than processing each one separately;
    so we merge them into one layer.
    """

    loss_type = LossType.CLASSIFICATION

    def __init__(self) -> None:
        self.output = None
        self.derivatives = None
        self.activation = Softmax()
        self.loss = CategoricalLoss()
        self.regularization = self.loss.regularization

    def forward(self, *, inputs: np.ndarray) -> None:
        """
        Forwarding the inputs to activation layer.
        """
        self.activation.forward(inputs)
        self.output = self.activation.output

    def calculate(self, y_true: np.ndarray, *_, **__) -> float:
        """
        Calculate the loss value. y_pred is same as `self.output` but for sake of consistency,
        we will accept *args and **kwargs as same but ignore them.
        This method must be used after a forward pass.
        """
        return self.loss.calculate(y_true, self.output)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Backward pass of the combined activation and loss functions.
        Backward: `y_pred - y_true`
        """
        check_shapes(y_true, y_pred)

        self.derivatives = y_pred.copy()
        # y_true == 1, so we need to subtract 1 from y_pred.
        self.derivatives[range(y_true.shape[0]), y_true] -= 1
        # Normalizing the derivatives.
        self.derivatives /= y_true.shape[0]
