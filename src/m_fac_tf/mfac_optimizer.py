# -*- coding: utf-8 -*-
""" Run F-MAC optimizer on dedicated examples"""

import tensorflow as tf

import tensorflow as tf
import tensorflow_datasets as tfds

loader = tfds.load("cifar10", as_supervised=True)
train, test = loader["train"], loader["test"]

model = tf.keras.applications.resnet.ResNet50()


if __name__ == "__main__":
    print(model)
