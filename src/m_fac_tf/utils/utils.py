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


