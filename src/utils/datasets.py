"""Function to load dataset and model."""

import tensorflow as tf
import tensorflow.keras.datasets as mnist
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.keras.utils import to_categorical

from src.optimizers.F_MFAC_ADAM import Adam_Mfac
from src.optimizers.F_MFAC_SGD import Mfac
from src.optimizers.MFAC import MFAC
from src.utils.helper_functions import build_resnet_20, build_resnet_32


def load_optimizer(name: str, params: dict):
    """Load the desired optimizer.

    Args:
        name: string for optimizer name
        params: dictionary of all necessary parameter values.

    Returns:
        Tensorflow Optimizer class

    Raises:
        UnknownNameError exception.
    """
    name = name.lower()
    if name == "adam":
        return tf.optimizers.Adam(**params)
    elif name == "sgd":
        return tf.optimizers.SGD(**params)
    elif name == "f-mfac-sgd":
        return Mfac(**params)
    elif name == "mfac":
        return MFAC(**params)
    elif name == "f-mfac-adam":
        return Adam_Mfac(**params)
    else:
        raise UnknownNameError(f"Given Optimizer name '{name}' is not implemeted.")  # noqa E501


class UnknownNameError(Exception):
    def __init__(self, name):
        self.name = name


def get_mnist_model_and_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation="softmax"))
    return (x_train, y_train, x_test, y_test, model)


def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))  # Resize to the input size of ResNet-50
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float
    image = (image - 0.5) * 2.0  # Normalize to the range [-1, 1]
    return image, label


def extract_text_and_labels(text, label):
    return text, tf.cast(label, tf.int32)


def get_dataset(name: str):
    """ """
    name = name.lower()
    if name == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode="fine")
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        return x_train, y_train, x_test, y_test

    elif name == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        return x_train, y_train, x_test, y_test

    elif name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        return x_train, y_train, x_test, y_test

    else:
        raise UnknownNameError(f"Requested dataset {name} not implemented.")


def get_model(name: str, n_classes, input_shape=None, top=False, weights=None):
    """ """
    name = name.lower()
    if name == "resnet20":
        model = build_resnet_20(input_shape=input_shape, num_classes=n_classes)
        return model

    elif name == "resnet32":
        model = build_resnet_32(input_shape=input_shape, num_classes=n_classes)
        return model

    elif name == "full_neural_network":
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(256, activation=tf.nn.relu),
                tf.keras.layers.Dense(256, activation=tf.nn.relu),
                tf.keras.layers.Dense(256, activation=tf.nn.relu),
                tf.keras.layers.Dense(256, activation=tf.nn.relu),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax),
            ]
        )
        return model
    else:
        raise UnknownNameError(f"Requested model {name} not implemented.")


if __name__ == "__main__":
    model, x_train, y_train, x_test, y_test = get_model_and_dataset("resnet50")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=3)
