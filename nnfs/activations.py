#! /usr/bin/env python3.10

"""
All necessary Activation functions.
`numpy` is required. -> https://numpy.org/
Compatible with python3.10+.
Mahyar@Mahyar24.com, Sat 23 Apr 2022.
"""

import abc

import numpy as np


class Activation(metaclass=abc.ABCMeta):
    """
    Abstract base class for activation functions.
    `forward` and `backward` methods must be implemented.
    """

    def __init__(self) -> None:
        self.inputs = None
        self.output = None
        self.derivatives = None

    @abc.abstractmethod
    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass of the activation function.
        """
        pass

    @abc.abstractmethod
    def backward(self, derivatives: np.ndarray) -> None:
        """
        Backward pass of the activation function.
        """
        pass


class ReLU(Activation):
    """
    Rectified Linear Unit activation function.
    Forward: `x > 0` -> `x` otherwise `0`
    Backward: `x > 0` -> `1` otherwise `0`
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass of the activation function.
        Forward: `x > 0` -> `x` otherwise `0`
        """
        # Remember the inputs for the backward pass
        self.inputs = inputs
        # Apply the activation function. ReLU is applied element-wise.
        self.output = np.maximum(0, inputs)

    def backward(self, derivatives: np.ndarray) -> None:
        """
        Backward pass of the activation function.
        Backward: `x > 0` -> `1` otherwise `0`
        """
        self.derivatives = derivatives.copy()
        # Apply the derivatives of the activation function.
        self.derivatives[self.inputs < 0] = 0


class LeakyReLU(Activation):
    """
    Leaky Rectified Linear Unit activation function.
    Forward: `x > 0` -> `x` otherwise `x * negative_slope`
    Backward: `x > 0` -> `1` otherwise `negative_slope`
    """

    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass of the activation function.
        Forward: `x > 0` -> `x` otherwise `x * negative_slope`
        """
        # Remember the inputs for the backward pass
        self.inputs = inputs
        # Apply the activation function. LeakyReLU is applied element-wise.
        self.output = np.where(inputs > 0, inputs, self.negative_slope * inputs)

    def backward(self, derivatives: np.ndarray) -> None:
        """
        Backward pass of the activation function.
        Backward: `x > 0` -> `1` otherwise `negative_slope`
        """
        self.derivatives = derivatives.copy()
        # Apply the derivatives of the activation function.
        self.derivatives[self.inputs < 0] *= self.negative_slope


class Sigmoid(Activation):
    """
    Sigmoid activation function.
    Forward: `1 / (1 + exp(-x))`
    Backward: `sig(x) * (1 - sig(x))`
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass of the activation function.
        Forward: `1 / (1 + exp(-x))`
        """
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, derivatives: np.ndarray) -> None:
        """
        Backward pass of the activation function.
        Backward: `sig(x) * (1 - sig(x))`
        """
        self.derivatives = derivatives * self.output * (1 - self.output)


class Linear(Activation):
    """
    Linear activation function, for regression.
    Forward: `x`
    Backward: `1`
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass of the activation function.
        Forward: `x`
        """
        self.output = inputs

    def backward(self, derivatives: np.ndarray) -> None:
        """
        Backward pass of the activation function.
        Backward: `1`
        """
        self.derivatives = derivatives.copy()


class Softmax(Activation):
    """
    Softmax activation function.
    Forward: `exp(x) / sum(exp(x))`
    Backward: `softmax(i) * (Kronecker_delta(i, j) - softmax(j))`
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass of the activation function.
        Forward: `exp(x) / sum(exp(x))`
        """
        exp_inputs = np.exp(
            inputs - np.max(inputs, axis=1, keepdims=True)
        )  # Normalizing for preventing overflow.
        self.output = exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)

    def backward(self, derivatives: np.ndarray) -> None:
        """
        Backward pass of the activation function.
        Backward: `softmax(i) * (Kronecker_delta(i, j) - softmax(j))`
        """
        self.derivatives = np.empty_like(derivatives)
        # For each example, for each class, we compute the derivative.
        # `for` loop makes it slow.
        for i, (output, derivatives) in enumerate(zip(self.output, derivatives)):
            output = output.reshape(-1, 1)
            jacobian = np.diagflat(output) - (output @ output.T)
            self.derivatives[i] = jacobian @ derivatives


class TanH(Activation):
    """
    Hyperbolic tangent activation function.
    Forward: `tanh(x)`
    Backward: `1 - tanh(x)**2`
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass of the activation function.
        Forward: `tanh(x)`
        """
        # Apply the activation function. ReLU is applied element-wise.
        self.output = np.tanh(inputs)

    def backward(self, derivatives: np.ndarray) -> None:
        """
        Backward pass of the activation function.
        Backward: `1 - tanh(x)**2`
        """
        self.derivatives = derivatives.copy()
        # Apply the derivatives of the activation function.
        self.derivatives *= 1 - self.output**2
