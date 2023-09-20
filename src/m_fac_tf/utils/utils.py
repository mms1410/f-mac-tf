# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd

class TrainingCallback(tf.keras.callbacks.Callback):
    """
    """
    def on_epoch_end(self, epoch, logs=None):
        weights = [w for w in self.model.trainable_weights if 'dense' in w.name and 'bias' in w.name]
        loss = self.model.total_loss
        optimizer = self.model.optimizer
        gradients = optimizer.get_gradients(loss, weights)

class ModelInspectionCallback(tf.keras.callbacks.Callback):
    """
    """
    def on_train_begin(self, epoch, logs=None):
        print("Inspecting model....\n")
        model = self.model
        model_dict = model.__dict__
        print(model.optimizer)

