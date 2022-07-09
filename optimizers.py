#! /usr/bin/env python3.10

"""
Implementation of various optimizers.
`numpy` is required. -> https://numpy.org/
Compatible with python3.10+.
Mahyar@Mahyar24.com, Sat 23 Apr 2022.
"""
import abc

import numpy as np

from nnfs.layer import Layer


class Optimizer(metaclass=abc.ABCMeta):
    """
    Abstract base class for optimizers.
    """

    @abc.abstractmethod
    def pre_update(self) -> None:
        """
        Pre-Update parameters.
        """
        pass

    @abc.abstractmethod
    def update(self, layer: Layer) -> None:
        """
        Update parameters.
        """
        pass

    @abc.abstractmethod
    def post_update(self) -> None:
        """
        Post-Update parameters.
        """
        pass


class Momentum(Optimizer):
    def __init__(
        self, learning_rate: float = 0.01, decay: float = 1e-5, beta: float = 0.1
    ) -> None:
        self.learning_rate = learning_rate
        self.decay = decay
        self.beta = beta
        self.iterations = 0
        self.current_learning_rate = learning_rate

    def pre_update(self) -> None:
        """
        Lower the learning rate.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate / (
                1 + (self.decay * self.iterations)
            )

    def update(self, layer: Layer) -> None:
        """
        Update parameters.
        """
        if not hasattr(layer, "v_dw"):
            layer.v_dw = np.zeros_like(layer.dw)
            layer.v_db = np.zeros_like(layer.db)

        layer.v_dw = self.beta * layer.v_dw + (1 - self.beta) * layer.dw
        layer.v_db = self.beta * layer.v_db + (1 - self.beta) * layer.db

        layer.weights -= self.current_learning_rate * layer.v_dw
        layer.biases -= self.current_learning_rate * layer.v_db

    def post_update(self) -> None:
        """
        Increase the iteration counter.
        """
        self.iterations += 1


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.01,
        decay: float = 1e-5,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-7,
    ) -> None:
        self.learning_rate = learning_rate
        self.decay = decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.iterations = 0
        self.current_learning_rate = learning_rate

    def pre_update(self) -> None:
        """
        Lower the learning rate.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate / (
                1 + (self.decay * self.iterations)
            )

    def update(self, layer: Layer) -> None:
        """
        Update parameters.
        """
        if not hasattr(layer, "v_dw"):
            layer.v_dw = np.zeros_like(layer.dw)
            layer.v_db = np.zeros_like(layer.db)
            layer.s_dw = np.zeros_like(layer.dw)
            layer.s_db = np.zeros_like(layer.db)

        layer.v_dw = self.beta_1 * layer.v_dw + (1 - self.beta_1) * layer.dw
        v_dw_corrected = layer.v_dw / (1 - (self.beta_1 ** (self.iterations + 1)))

        layer.v_db = self.beta_1 * layer.v_db + (1 - self.beta_1) * layer.db
        v_db_corrected = layer.v_db / (1 - (self.beta_1 ** (self.iterations + 1)))

        layer.s_dw = self.beta_2 * layer.s_dw + (1 - self.beta_2) * (layer.dw**2)
        s_dw_corrected = layer.s_dw / (1 - (self.beta_2 ** (self.iterations + 1)))

        layer.s_db = self.beta_2 * layer.s_db + (1 - self.beta_2) * (layer.db**2)
        s_db_corrected = layer.s_db / (1 - (self.beta_2 ** (self.iterations + 1)))

        w_update = v_dw_corrected / (np.sqrt(s_dw_corrected) + self.eps)
        b_update = v_db_corrected / (np.sqrt(s_db_corrected) + self.eps)

        layer.weights -= self.current_learning_rate * w_update
        layer.biases -= self.current_learning_rate * b_update

    def post_update(self) -> None:
        """
        Increase the iteration counter.
        """
        self.iterations += 1
