"""Model builder."""
from typing import Tuple

import keras
import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    ReLU,
)
from tensorflow.keras.models import Model


def residual_block(x: tf.keras.layers, filters: int):
    # Define a single residual block
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x


def build_resnet_20(input_shape, num_classes):
    # Define the input layer
    input_layer = tf.keras.layers.Input(shape=input_shape)

    # Initial convolution and max-pooling
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Stack residual blocks
    num_blocks = 6  # 6 residual blocks for a total of 20 layers
    for _ in range(num_blocks):
        x = residual_block(x, 16)

    # Global average pooling and final dense layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=input_layer, outputs=x, name='resnet20')

    return model

def build_resnet_32(input_shape: Tuple[(int, int, int)], num_classes: int) -> keras.src.engine.functional.Functional:  # noqa E501
    """Build Resnet 32 model.

    Build a renet32 model based on the desired input shape and classes.

    Args:
        input_shape: Tuple of form (int, int, int).
        num_classes: integer number.

    Returns:
        keras model
    """
    # Define the input layer
    input_layer = tf.keras.layers.Input(shape=input_shape)

    # Initial convolution and max-pooling
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Stack residual blocks
    num_blocks = 10  # 10 residual blocks for a total of 32 layers
    for _ in range(num_blocks):
        x = residual_block(x, 16)

    # Global average pooling and final dense layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=input_layer, outputs=x, name='resnet32')

    return model


def wide_residual_block(x, filters, stride):
    identity = x

    # First convolution
    x = Conv2D(filters, (3, 3), strides=(stride, stride), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second convolution
    x = Conv2D(filters, (3, 3), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    # If the stride is greater than 1, the input shape needs to be adjusted.
    if stride > 1:
        identity = Conv2D(filters, (1, 1), strides=(stride, stride))(identity)

    x = Add()([x, identity])
    x = ReLU()(x)

    return x


# Define the WideResNet-40-2 model
def create_wideresnet40_2(input_shape=(32, 32, 3), num_classes=10):
    input_tensor = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), padding='same')(input_tensor)

    num_blocks_list = [3, 3, 3]  # 3 blocks with 16, 32, 64 filters
    filters = 16

    for num_blocks in num_blocks_list:
        x = wide_residual_block(x, filters, stride=1)
        for _ in range(num_blocks - 1):
            x = wide_residual_block(x, filters, stride=1)
        filters *= 2

    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=x, name='wideresnet40_2')
    return model


if __name__ == "__main__":
    input_shape = (32, 32, 3)
    num_classes = 10
    model = build_resnet_32(input_shape, num_classes)
    print(type(model))