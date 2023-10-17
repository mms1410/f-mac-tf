"""Function to load dataset and model."""

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import tensorflow.keras.datasets as mnist
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications import ResNet50
from keras.applications.mobilenet import MobileNet
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models


def load_optimizer(name: str, params: dict):
    """Load the desired optimizer.
    
    Args:
        name: string for optimizer name
        params: dictionary of all necessary parameter values.
    
    Returns:
        Tensosrflow Optimizer class

    Raises:
        UnknownNameError exception.
    """
    if name == "adam":
        return tf.optimizers.Adam(**params)
    if name == "sgd":
        return tf.optimizers.SGD(**params)
    else:
        raise UnknownNameError(f"Given Optimizer name '{name}' isn ot implemeted.")  # noqa E501


class UnknownNameError(Exception):
    def __init__(self, name):
        self.name = name


def get_mnist_model_and_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(10, activation='softmax'))
    return (x_train, y_train, x_test, y_test, model)


def preprocess_image(image, label):
    image = tf.image.resize(image, (224, 224))  # Resize to the input size of ResNet-50
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float
    image = (image - 0.5) * 2.0  # Normalize to the range [-1, 1]
    return image, label

def extract_text_and_labels(text, label):
    return text, tf.cast(label, tf.int32)

def get_dataset(name: str):
    """
    """
    name = name.lower()
    if name == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        y_train = to_categorical(y_train, 100)
        y_test = to_categorical(y_test, 100)
        return x_train, y_train, x_test, y_test
    elif name == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        return x_train, y_train, x_test, y_test
    elif name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        return train_dataset, test_dataset, None
    elif name == "imagenet":
        dataset, info = tfds.load("imagenet2012", with_info=True,
                                  as_supervised=True,
                                  download=False)
        train_dataset, test_dataset = dataset["train"], dataset["test"]
        # Extract features and labels
        x_train, y_train = zip(*[(image, label) for image, label in train_dataset])
        x_test, y_test = zip(*[(image, label) for image, label in test_dataset])

        #  Convert to TensorFlow tensors
        x_train = tf.convert_to_tensor(x_train)
        x_test = tf.convert_to_tensor(x_test)
        y_train = tf.convert_to_tensor(y_train)
        y_test = tf.convert_to_tensor(y_test)
        return x_train, y_train, x_test, y_test
    elif name == "imbd":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb
        train_dataset, test_dataset = dataset["train"], dataset["test"]
        train_dataset = train_dataset.map(extract_text_and_labels)
        test_dataset = test_dataset.map(extract_text_and_labels)
        return train_dataset, test_dataset, info
    else:
        raise UnknownNameError(f"Requested dataset {name} not implemented.")


def get_model(name: str, n_classes, input_shape=None, top=False, weights=None):
    """
    """
    name = name.lower()
    if name == "resnet50":
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        return model
    elif name == "resnet101":
        base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        #model = keras.applications.ResNet101(weights=weights,
        #                                            include_top=top,
        #                                            input_shape=input_shape,
        #                                           classes=n_classes)
        #base_model = tf.keras.applications.resnet.ResNet101(weights=weights,
        #                                                    include_top=top,
        #                                                    input_shape=input_shape)
        #inputs = keras.Input(shape=input_shape)
        #x = base_model(inputs)
        #x = keras.layers.GlobalAveragePooling2D()(x)
        #outputs = keras.layers.Dense(n_classes)(x)
        #model = keras.Model(input, outputs)
        return model
    elif name == "resnet152":
        model = tf.keras.applications.ResNet152(weights=weights,
                                                include_top=top,
                                                input_shape=input_shape)
        #model = tf.keras.applications.resnet.ResNet152(weights=weights,
        #                                               include_top=top,
        #                                               input_shape=input_shape)
        inputs = tf.keras.Input(shape=input_shape)
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(n_classes)(x)
        model = tf.keras.Model(input, outputs)
        return model
    else:
        raise UnknownNameError(f"Requested model {name} not implemented.")

def get_model_and_dataset(model_name):
    # Dictionary mapping model names to corresponding datasets
    model_dataset_map = {
        "bert": "imbd",
        "resnet101": "cifar10",
        "resnet152": "cifar10",
        "resnet50": "cifar10",
        "inceptionv3": "imagenet",
        "mobilenetv2": "imagenet",
        "inceptionresnetv2": "imagenet",
        "mobilenetv1": "ImageNet",
    }
    # Check if the provided model name is in the dictionary
    if model_name in model_dataset_map:
        dataset_name = model_dataset_map[model_name]
        x_train, y_train, x_test, y_test = get_dataset(dataset_name)
        input_shape = x_train.shape[1:]
        n_classes = x_train.shape[-1]
        model = get_model(model_name,
                          input_shape=input_shape,
                          top=True,
                          n_classes=n_classes)
        return model, x_train, y_train, x_test, y_test
    else:
        raise UnknownNameError(f"Model name {model_name} is not supported or cannot be processed.")


if __name__ == "__main__":
    model, x_train, y_train, x_test, y_test = get_model_and_dataset("resnet50")
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=3)

