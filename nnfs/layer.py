#! /usr/bin/env python3.10

"""
Neural Network Layer implementations.
`numpy` is required. -> https://numpy.org/
Compatible with python3.10+.
Mahyar@Mahyar24.com, Sat 23 Apr 2022.
"""
import numpy as np


class Layer:
    """
    Neural Network Layer.
    """

    def __init__(
        self,
        num_inputs: int,
        num_neurons: int,
        *,
        w_l1: float = 0.0,
        b_l1: float = 0.0,
        w_l2: float = 0.0,
        b_l2: float = 0.0,
    ) -> None:
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.weights = np.random.randn(num_inputs, num_neurons) * 0.01
        self.biases = np.zeros((1, num_neurons))

        # Regularization
        self.w_l1 = w_l1
        self.b_l1 = b_l1
        self.w_l2 = w_l2
        self.b_l2 = b_l2

        self.inputs = None
        self.output = None
        self.dw = None
        self.db = None
        self.derivatives = None

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass of the layer.
        Forward: `W . X + b`
        """
        assert len(inputs.shape) == 2, f"Inputs ({inputs.shape=}) must be a 2D array."
        assert (
            inputs.shape[1] == self.num_inputs
        ), f"Inputs ({inputs.shape=}) must have the same number of columns as the layer. ({self.num_inputs=})"

        # Remember the inputs for the backward pass.
        self.inputs = inputs
        self.output = inputs @ self.weights + self.biases

    def backward(self, derivatives: np.ndarray) -> None:
        """
        Backward pass of the layer.
        Backward: `dW = inputs.T . derivatives, db = derivatives.T.sum(), derivatives = derivatives @ W.T`
        """
        self.dw = self.inputs.T @ derivatives
        self.db = derivatives.sum(axis=0, keepdims=True)
        self.derivatives = derivatives @ self.weights.T

        # L1 Regularization
        if self.w_l1 > 0:
            d_w_l1 = np.ones_like(self.weights)
            d_w_l1[self.weights < 0] = -1
            self.dw += self.w_l1 * d_w_l1
        if self.b_l1 > 0:
            d_b_l1 = np.ones_like(self.biases)
            d_b_l1[self.biases < 0] = -1
            self.db += self.b_l1 * d_b_l1
        # L2 Regularization
        if self.w_l2 > 0:
            d_w_l2 = 2 * self.w_l2 * self.weights
            self.dw += d_w_l2
        if self.b_l2 > 0:
            d_b_l2 = 2 * self.b_l2 * self.biases
            self.db += d_b_l2


class Dropout:
    """
    Dropout layer.
    """

    def __init__(self, drop_rate: float) -> None:
        self.rate = 1 - drop_rate

        self.output = None
        self.derivatives = None
        self.mask = None

    def forward(self, inputs: np.ndarray) -> None:
        """
        Forward pass of the layer.
        Change the values of the inputs to 0 with a probability of `self.rate`.
        """
        # Normalizing the mask to keep the same scale with the inputs.
        self.mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        # Element-wise multiplication of the inputs and the mask.
        self.output = inputs * self.mask

    def backward(self, derivatives):
        """
        Backward pass of the layer.
        """
        self.derivatives = derivatives * self.mask
