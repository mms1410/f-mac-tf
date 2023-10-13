"""Function to load dataset and model."""

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from src.m_fac_tf.models import build_resnet_20, build_resnet_32


def load_dataset(name: str):
    """
    """
    if name.lower() == "cifar10":
        (train_dataset, test_dataset) = tfds.load(
            "cifar10",
            split=["train", "test"],
            as_supervised=True)
    elif name.lower() == "cifar100":
        (train_dataset, test_dataset)  = tfds.load(
            "cifar100",
            split=["train", "test"],
            as_supervised=True)
    elif name.lower() == "squadv2":
        pass
    elif name.lower() == "glue":
        pass
    else:
        pass  # ToDo: raise exception


def load_model(name: str):
    """Loads model and data"""

    if name.lower() == "resnet50":
        input_shape = (224, 224, 3)
        num_classes = 1000 
        model = tf.keras.applications.ResNet50(
             include_top=True,
             weights="imagenet",
             input_shape=input_shape,
             classes=num_classes)
    elif name.lower() == "resnet20":
        input_shape = (32, 32, 3)
        num_classes = 10
        model = build_resnet_20(input_shape, num_classes)
    elif name.lower() == "resnet32":
        input_shape = (32, 32, 3)
        num_classes = 10
        model = build_resnet_32(input_shape, num_classes)
    elif name.lower() == "inceptionv3":
        input_shape = (224, 224, 3)
        num_classes = 1000
        model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=True,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=1000,
            classifier_activation="softmax"
            )
    elif name.lower() == "mobilenetv1":
        input_shape = (224, 224, 3)
        num_classes = 100
        
        model = tf.keras.applications.mobilenet.MobileNet(
            input_shape=input_shape,
            classes=num_classes)
    elif name.lower() == "imagenet":
        input_shape = (224, 224, 3)
        num_classes = 100
        model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"
        model = tf.keras.Sequential([hub.KerasLayer(model_url,
                                                    output_shape=[1000])])
    else:
        pass  # ToDo: Raise Exception
    return model