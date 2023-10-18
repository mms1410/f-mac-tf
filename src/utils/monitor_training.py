"""Callbacks, loggers, profilers etc. to track experiment results."""
import tensorflow as tf
import logging
import datetime
import time


class ModelInspectionCallback(tf.keras.callbacks.Callback):
    """
    """
    # def on_train_batch_end(self, batch, epoch, logs=None):
    #    print(f"\nCall on train_batch_end\n")
    #    model = self.model
    #    optimizer = model.optimizer

    def on_epoch_end(self, epoch, logs=None):
        print("\nCall on_epoch_end\n")
        model = self.model
        optimizer = model.optimizer
        print(f"MATRIX FIFO COUNTER: {optimizer.GradFifo.counter}")
        print(f"TEST SLOT 1: {optimizer.test_slot_1}")
        print(f"TEST SLOT 2: {optimizer.test_slot_2}")


class MFACresetCallback(tf.keras.callbacks.Callback):
    """
    """
    def on_epoch_end(self, epoch, logs=None):
        model = self.model
        optimizer = model.optimizer
        optimizer.GradFifo.reset()

class ProfilerCallback(tf.keras.callbacks.Callback):
    def __init__(self, log_file_path):
        super(ProfilerCallback, self).__init__()
        self.log_file_path = log_file_path
        self.start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        end_time = datetime.datetime.now()
        elapsed_time = end_time - self.start_time
        message = f"Epoch {epoch + 1} elapsed time: {elapsed_time}"
        self._write_to_log(message)

    def on_train_begin(self, logs=None):
        self.profile = tf.profiler.experimental.Profiler(self.log_file_path)

    def on_train_batch_end(self, batch, logs=None):
        if batch % 10 == 0:  # Adjust the frequency of profiling
            tf.profiler.experimental.start(logdir=self.log_file_path)
            tf.profiler.experimental.stop()

    def on_train_end(self, logs=None):
        self.profile = None

    def _write_to_log(self, message):
        logging.basicConfig(filename=self.log_file_path, level=logging.INFO)
        logging.info(message)

class TimeCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # Calculate the elapsed time in seconds
        elapsed_time = time.time() - self.start_time
        logs["elapsed_time"] = elapsed_time