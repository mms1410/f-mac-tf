"""Model builder."""
import tensorflow as tf
from typing import Tuple
import keras


def residual_block(x:tf.keras.layers, filters:int):
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


if __name__ == "__main__":
    input_shape = (32, 32, 3)
    num_classes = 10
    model = build_resnet_32(input_shape, num_classes)
    print(type(model))