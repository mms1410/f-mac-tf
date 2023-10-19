"""Callbacks, loggers, profilers etc. to track experiment results."""
import csv
import time

import tensorflow as tf


class CustomCSVLogger(tf.keras.callbacks.Callback):
    def __init__(self, filename):
        super(CustomCSVLogger, self).__init__()
        self.filename = filename
        self.csv_file = open(filename, mode="w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["epoch", "time", "loss", "accuracy", "val_loss", "val_accuracy"])

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        row = [
            epoch,
            elapsed_time,
            logs["loss"],
            logs["accuracy"],
            logs["val_loss"],
            logs["val_accuracy"],
        ]
        self.csv_writer.writerow(row)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
