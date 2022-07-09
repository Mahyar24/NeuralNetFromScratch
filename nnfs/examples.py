#! /usr/bin/env python3.10

"""
Different cases of using the nnfs.
`numpy` is required. -> https://numpy.org/
Compatible with python3.10+.
Mahyar@Mahyar24.com, Sat 23 Apr 2022.
"""

from nnfs.activations import Linear, ReLU, Sigmoid, Softmax
from nnfs.layer import Dropout, Layer
from nnfs.loss import BinaryLoss, CategoricalLoss, MSELoss, SoftmaxLoss
from nnfs.metrics import Accuracy, ExplainedVariance
from nnfs.model import Model
from nnfs.optimizers import Adam


def classification_SoftmaxLoss(X_train, y_train, X_test, y_test):
    """
    Classification model with SoftmaxLoss and Adam optimizer.
    3 classes and 2 features.
    """
    model = Model(loss=SoftmaxLoss(), optimizer=Adam(), metric=Accuracy())
    model.add(Layer(2, 64, w_l2=5e-4, b_l2=5e-4))
    model.add(ReLU())
    model.add(Layer(64, 3))
    model.fit(X_train, y_train, epochs=1_000, batch_size=512)
    validation_accuracy = Accuracy.evaluate(y_test, model.predict(X_test))
    print(f"Validation Accuracy: {validation_accuracy:.2%}")
    return model


def classification_Softmax_Categorical(X_train, y_train, X_test, y_test):
    """
    Classification model with Softmax Activation and CategoricalLoss (separately) and Adam optimizer.
    3 classes and 2 features.
    """
    model = Model(loss=CategoricalLoss(), optimizer=Adam(), metric=Accuracy())
    model.add(Layer(2, 64, w_l2=5e-4, b_l2=5e-4))
    model.add(ReLU())
    model.add(Layer(64, 3))
    model.add(Softmax())
    model.fit(X_train, y_train, epochs=1_000, batch_size=512)
    validation_accuracy = Accuracy.evaluate(y_test, model.predict(X_test))
    print(f"Validation Accuracy: {validation_accuracy:.2%}")
    return model


def classification_Sigmoid_BinaryLoss(X_train, y_train, X_test, y_test):
    """
    Classification model with Sigmoid Activation and BinaryLoss and Adam optimizer.
    2 classes and 2 features.
    """
    model = Model(loss=BinaryLoss(), optimizer=Adam(), metric=Accuracy())
    model.add(Layer(2, 64, w_l2=5e-4, b_l2=5e-4))
    model.add(ReLU())
    model.add(Layer(64, 1))
    model.add(Sigmoid())
    model.fit(X_train, y_train, epochs=1_000, batch_size=512)
    validation_accuracy = Accuracy.evaluate(y_test, model.predict(X_test))
    print(f"Validation Accuracy: {validation_accuracy:.2%}")
    return model


def regression_Linear_MSE(X_train, y_train, X_test, y_test):
    """
    Regression model with Linear activation and MSE loss and Adam optimizer.
    1 feature.
    """
    model = Model(loss=MSELoss(), optimizer=Adam(), metric=ExplainedVariance())
    model.add(Layer(1, 64, w_l2=5e-4, b_l2=5e-4))
    model.add(ReLU())
    model.add(Layer(64, 64))
    model.add(ReLU())
    model.add(Layer(64, 1))
    model.add(Linear())
    model.fit(X_train, y_train, epochs=1_000, batch_size=512)
    validation_accuracy = ExplainedVariance.evaluate(y_test, model.predict(X_test))
    print(f"Validation Accuracy: {validation_accuracy:.2%}")
    return model


def classification_SoftmaxLoss_Dropout(X_train, y_train, X_test, y_test):
    """
    Classification model with SoftmaxLoss and Adam optimizer.
    3 classes and 2 features with Dropout.
    """
    model = Model(loss=SoftmaxLoss(), optimizer=Adam(), metric=Accuracy())
    model.add(Layer(2, 512))
    model.add(ReLU())
    model.add(Dropout(0.05))
    model.add(Layer(512, 3))
    model.fit(X_train, y_train, epochs=1_000, batch_size=512)
    validation_accuracy = Accuracy.evaluate(y_test, model.predict(X_test))
    print(f"Validation Accuracy: {validation_accuracy:.2%}")
    return model


def regression_Linear_MSE_Dropout(X_train, y_train, X_test, y_test):
    """
    Regression model with Linear activation and MSE loss and Adam optimizer.
    1 feature with Dropout.
    """
    model = Model(loss=MSELoss(), optimizer=Adam(), metric=ExplainedVariance())
    model.add(Layer(1, 512))
    model.add(ReLU())
    model.add(Dropout(0.05))
    model.add(Layer(512, 512))
    model.add(ReLU())
    model.add(Layer(512, 1))
    model.add(Linear())
    model.fit(X_train, y_train, epochs=200, batch_size=512)
    validation_accuracy = ExplainedVariance.evaluate(y_test, model.predict(X_test))
    print(f"Validation Accuracy: {validation_accuracy:.2%}")
    return model


def classification_SoftmaxLoss_MNIST():
    from tensorflow.keras.datasets import mnist

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255

    model = Model(loss=SoftmaxLoss(), optimizer=Adam(), metric=Accuracy())
    model.add(Layer(28 * 28, 512))
    model.add(ReLU())
    model.add(Layer(512, 10))
    model.fit(train_images, train_labels, epochs=10, batch_size=512)
    validation_accuracy = Accuracy.evaluate(test_labels, model.predict(test_images))
    print(f"Validation Accuracy: {validation_accuracy:.2%}")
    return model


if __name__ == "__main__":
    # It will not work here, because this package name is `nnfs` too.
    # Install the original package via `python3.10 -m pip install nnfs`
    # and use it inside a Jupyter notebook.
    from nnfs.datasets import sine_data, spiral_data

    # Make data
    (X_train_clf_3, y_train_clf_3), (X_test_clf_3, y_test_clf_3) = spiral_data(
        samples=500, classes=3
    ), spiral_data(samples=50, classes=3)
    (X_train_clf_2, y_train_clf_2), (X_test_clf_2, y_test_clf_2) = spiral_data(
        samples=500, classes=2
    ), spiral_data(samples=50, classes=2)
    X_reg_1, y_reg_1 = sine_data(3100)
    y_reg_1 = y_reg_1.reshape(-1)
    X_train_reg_1 = X_reg_1[:3000, :].copy()
    y_train_reg_1 = y_reg_1[:3000].copy()
    X_test_reg_1 = X_reg_1[3000:3100, :].copy()
    y_test_reg_1 = y_reg_1[3000:3100].copy()

    # Test.
    print("classification_SoftmaxLoss: ")
    classification_SoftmaxLoss(X_train_clf_3, y_train_clf_3, X_test_clf_3, y_test_clf_3)

    print("classification_Softmax_Categorical: ")
    classification_Softmax_Categorical(
        X_train_clf_3, y_train_clf_3, X_test_clf_3, y_test_clf_3
    )

    print("classification_Sigmoid_BinaryLoss: ")
    classification_Sigmoid_BinaryLoss(
        X_train_clf_2, y_train_clf_2, X_test_clf_2, y_test_clf_2
    )

    print("regression_Linear_MSE: ")
    regression_Linear_MSE(X_train_reg_1, y_train_reg_1, X_test_reg_1, y_test_reg_1)

    print("classification_SoftmaxLoss_Dropout: ")
    classification_SoftmaxLoss_Dropout(
        X_train_clf_3, y_train_clf_3, X_test_clf_3, y_test_clf_3
    )

    print("regression_Linear_MSE_Dropout: ")
    regression_Linear_MSE_Dropout(
        X_train_reg_1, y_train_reg_1, X_test_reg_1, y_test_reg_1
    )

    print("MNIST: ")
    classification_SoftmaxLoss_MNIST()
