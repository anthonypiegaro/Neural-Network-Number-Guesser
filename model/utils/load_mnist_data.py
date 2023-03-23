from typing import Tuple

from keras.datasets import mnist
from numpy import ndarray

def load_mnist_data() -> Tuple[ndarray, ...]:
    """
    Load MNIST data into training and test set
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return (X_train, y_train, X_test, y_test)
