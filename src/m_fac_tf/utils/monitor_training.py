"""Callbacks, loggers, profilers etc. to track experiment results."""
import tensorflow as tf
from typing import Tuple, List
import pandas as pd
 
class ModelInspectionCallback(tf.keras.callbacks.Callback):
    """
    """
    def on_train_batch_end(self, batch, epoch, logs=None):
        print(f"\nCall on train_batch_end\n")
        model = self.model
        optimizer = model.optimizer
        print(optimizer.GradFifo.values)
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nCall on_epoch_end\n")
        model = self.model
        optimizer = model.optimizer
        print("Test Slot 1:\n")
        for grad in optimizer.test_slot_1:
            print(grad)