# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import sys

class TrainingCallback(tf.keras.callbacks.Callback):
    """
    """
    def on_epoch_end(self, epoch, logs=None):
        model = self.model
        logs = logs or {}
        print("\n TESTCOUNTER: ", model.optimizer.test_counter)
        print("\n GRAD:        ", model.optimizer.base_grad)
        print("\n               ", type(model.optimizer.base_grad))
        print("\n               ", model.optimizer.base_grad.shape)


class ModelInspectionCallback(tf.keras.callbacks.Callback):
    """
    """
    def on_train_begin(self, epoch, logs=None):
        print("Inspecting model....\n")
        model = self.model
        model_dict = model.__dict__
        print(model.optimizer)


def get_simple_raw_model(input_shape, target_size):
    """
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=16, input_shape=input_shape, kernel_size=(3, 3),
               activation='relu', name='conv_1'),
        tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', name='conv_2'),
        tf.keras.layers.MaxPooling2D(pool_size=(4, 4), name='pool_1'),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(units=32, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(units=target_size, activation='softmax', name='dense_2')])
    return model