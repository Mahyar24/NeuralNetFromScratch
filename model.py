#! /usr/bin/env python3.10

"""
Implementation of Model.
`numpy` is required. -> https://numpy.org/
Compatible with python3.10+.
Mahyar@Mahyar24.com, Sat 23 Apr 2022.
"""
from __future__ import annotations

import copy
import pickle
from statistics import fmean
from typing import Iterator, Optional, Union

import numpy as np

# In Jupyter Notebook, try: `from tqdm.notebook import tqdm` instead.
from tqdm import tqdm

from nnfs.activations import Activation, Sigmoid
from nnfs.layer import Dropout, Layer
from nnfs.loss import BinaryLoss, Loss, LossType, SoftmaxLoss
from nnfs.metrics import Metric
from nnfs.optimizers import Optimizer


class Model:
    """
    Neural Network Model
    """

    EVALUATION_MODE_NO_FORWARD_LAYERS = (Dropout,)

    def __init__(
        self, *, loss: Union[Loss, SoftmaxLoss], optimizer: Optimizer, metric: Metric
    ) -> None:
        self.layers = []
        # Layers that will propagate in evaluation mode too.
        self.layers_eval = []
        if isinstance(loss, Loss) or isinstance(loss, SoftmaxLoss):
            self.loss = loss
        else:
            raise ValueError("loss must be an instance of Loss or SoftmaxLoss.")
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError("optimizer must be an instance of Optimizer.")
        if isinstance(metric, Metric):
            self.metric = metric
        else:
            raise ValueError("metric must be an instance of Metric.")
        self.data_ = {}

    def add(self, layer: Union[Layer, Activation, Layer]) -> None:
        """
        Add a layer to the model.
        """
        assert (
            isinstance(layer, Layer)
            or isinstance(layer, Activation)
            or isinstance(layer, Dropout)
        ), "layer must be an instance of Layer or Activation."

        if len(self.layers) == 0 and not isinstance(layer, Layer):
            raise ValueError("First Layer must be `Linear Layer`.")

        self.layers.append(layer)
        if not any(
            isinstance(layer, no_eval_layer)
            for no_eval_layer in self.EVALUATION_MODE_NO_FORWARD_LAYERS
        ):
            self.layers_eval.append(layer)

    @staticmethod
    def check_dimensions(X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        assert len(X.shape) == 2, f"X must be a 2D array. Got {X.shape}"
        if y is not None:
            assert len(y.shape) == 1, f"y must be a 1D array. Got {y.shape}"
            assert (
                X.shape[0] == y.shape[0]
            ), f"X and y must have the same number of rows. {X.shape[0]=} != {y.shape[0]=}"

    def forward(self, X: np.ndarray, evaluate_mode: bool = False) -> np.ndarray:
        """
        Forward propagation.
        """
        if evaluate_mode:
            # Evaluation mode; we should not propagate no-eval layers.
            layers = self.layers_eval
        else:
            layers = self.layers

        for i, layer in enumerate(layers):
            if i == 0:
                layer.forward(X)
            else:
                layer.forward(layers[i - 1].output)

        output = layers[-1].output

        # SoftmaxLoss is an edge case.
        if isinstance(self.loss, SoftmaxLoss):
            self.loss.forward(inputs=output)
            return self.loss.output

        return output

    def backward(self, y: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Backward propagation.
        """
        self.loss.backward(y, y_pred)
        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                layer.backward(self.loss.derivatives)
            else:
                layer.backward(self.layers[-i].derivatives)

        return self.layers[0].derivatives

    def optimize(self) -> None:
        """
        Optimize the neural layers of the model.
        """
        self.optimizer.pre_update()
        # Reversing just for demonstration purposes.
        for layer in reversed(self.layers):
            if isinstance(layer, Layer):
                self.optimizer.update(layer)
        self.optimizer.post_update()

    def record_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: list[float],
        accuracies: list[float],
        losses: list[float],
        regularization_losses: list[float],
    ) -> None:
        """
        Record data. This method should only use adder a forward propagation.
        """
        lr.append(self.optimizer.current_learning_rate)
        accuracies.append(self.metric.evaluate(y, self.predict(X)))
        losses.append(self.loss.calculate(y, self.layers[-1].output))
        regularization_losses.append(
            sum(
                self.loss.regularization(layer)
                for layer in self.layers
                if isinstance(layer, Layer)
            )
        )

    @staticmethod
    def shuffle_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Shuffle data.
        """
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        return X[indices], y[indices]

    def generate_batch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """
        Generate a batch of data.
        """
        if batch_size is not None:
            assert (
                batch_size <= X.shape[0]
            ), f"{batch_size=} must be smaller than {X.shape[0]=}"
        else:
            # If batch_size is None, use the whole data.
            batch_size = X.shape[0]

        if shuffle:
            X, y = self.shuffle_data(X, y)

        for i in range(0, X.shape[0], batch_size):
            yield X[i : i + batch_size], y[i : i + batch_size]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs=1_000,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
    ) -> None:
        """
        Fit the model.
        """
        self.check_dimensions(X, y)

        # Adding progress bar with tqdm.
        for epoch in (
            bar := tqdm(
                range(1, epochs + 1),
                desc="Iterating ...",
                total=epochs,
                bar_format="{desc}: {bar} {n_fmt}/{total_fmt} {percentage:3.0f}%",
            )
        ):
            # Initiate some data for recording for each epoch.
            lr: list[float] = []
            accuracies: list[float] = []
            losses: list[float] = []
            regularization_losses: list[float] = []

            # Optimize for a mini-batch.
            for X_batch, y_batch in self.generate_batch(X, y, batch_size, shuffle):
                # Forward propagation
                output = self.forward(X_batch)
                # Record data
                self.record_data(
                    X_batch, y_batch, lr, accuracies, losses, regularization_losses
                )
                # Backward propagation
                self.backward(y_batch, output)
                # Optimize
                self.optimize()

            # Record data
            self.data_[epoch] = {
                "lr": fmean(lr),
                "accuracy": fmean(accuracies),
                "loss": fmean(losses),
                "regularization_loss": fmean(regularization_losses),
            }

            # Print info in formatted manner.
            info_str = " ".join([f"{k}: {v:.4f}" for k, v in self.data_[epoch].items()])
            bar.set_description(info_str)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.
        """
        self.check_dimensions(X)

        if self.loss.loss_type == LossType.REGRESSION:
            raise NotImplementedError(
                "For prediction regression problem, use `predict` method."
            )

        output = self.forward(X, evaluate_mode=True)

        # Sigmoid
        if isinstance(self.layers[-1], Sigmoid) and not isinstance(
            self.loss, SoftmaxLoss
        ):
            return np.hstack((1 - output, output))
        # Softmax
        else:
            if output.shape[1] == 1:  # Using Softmax inappropriately.
                return output.reshape(-1)
            return output

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output of the model.
        """
        self.check_dimensions(X)

        if self.loss.loss_type == LossType.CLASSIFICATION:
            return self.predict_proba(X).argmax(axis=1)
        return self.forward(X, evaluate_mode=True).reshape(-1)

    @staticmethod
    def load(path: str) -> Model:
        """
        Load the model.
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def cleanup(self) -> Model:
        """
        Removing redundant attributes to save space.
        """

        new_model = copy.deepcopy(self)

        def cleanup_activation(activation: Activation) -> None:
            """
            Cleanup activation layers.
            """
            activation.inputs = None
            activation.output = None
            activation.derivatives = None

        def cleanup_layer(layer: Layer) -> None:
            """
            Cleanup layers.
            Optimizer caches will not be removed here. (because it will speed up further training runs)
            """
            layer.inputs = None
            layer.output = None
            layer.derivatives = None

        def cleanup_dropout(dropout: Dropout) -> None:
            """
            Cleanup dropout layers.
            """
            dropout.mask = None
            dropout.output = None
            dropout.derivatives = None

        def cleanup_loss(loss: Loss) -> None:
            """
            cleanup model's loss.
            """
            loss.derivatives = None
            if hasattr(loss, "activation") and isinstance(loss.activation, Activation):
                cleanup_activation(loss.activation)

        new_model.data_ = {}

        cleanup_loss(new_model.loss)

        for i in range(len(new_model.layers)):
            layer = new_model.layers[i]
            if isinstance(layer, Activation):
                cleanup_activation(layer)
            elif isinstance(layer, Dropout):
                cleanup_dropout(layer)
            elif isinstance(layer, Layer):
                cleanup_layer(layer)

        return new_model

    def save(self, path: str, cleanup: bool = True) -> None:
        """
        Save the model. if cleanup is True, it will remove redundant attributes.
        """
        # Purge records and non-essential attributes.
        if cleanup:
            pickling_model = self.cleanup()
        else:
            pickling_model = self

        with open(path, "wb") as f:
            pickle.dump(pickling_model, f, protocol=pickle.HIGHEST_PROTOCOL)
