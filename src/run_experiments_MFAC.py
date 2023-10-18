from datetime import datetime as datetime
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import CSVLogger
from src.utils.monitor_training import TimeCallback
from src.optimizers.MFAC import Mfac
from tensorflow.keras.optimizers.experimental import SGD as SGD
import tensorflow.keras.datasets.mnist as mnist
from src.utils.helper_functions import write_results_to_plot
from src.utils.datasets import get_dataset, get_model


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[3]
    epochs = 30
    m = 512
    damp = 1e-8
    batch_size = 1000
    optimizer = Mfac(m=m, damp=damp)
    data_name = "cifar10"
    model_name = "resnet20"
    n_classes = 10
    input_shape = (32, 32, 3)
    x_train, y_train, x_test, y_test, model = get_dataset(data_name)
    model = get_model(model_name, n_classes, input_shape)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

    (training_data, training_labels), (test_data, test_labels) = mnist.load_data()
    training_data, test_data = training_data / 255, test_data / 255

    if optimizer.name == "MFAC":
        additional_log_data = f"_m{m}_damp{damp}"
    else:
        additional_log_data = ""
    name = f"training_log_{optimizer.name}_batch{batch_size}" + additional_log_data  # noqa E501
    csv_name = "logs/" + name + ".csv"
    asset_name = "assets/" + name + ".png"
    csv_logger = CSVLogger(csv_name, separator=",", append=False)  # noqa E501
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"],
                  run_eagerly=True)
    model.fit(training_data, training_labels, epochs=epochs,
              batch_size=batch_size,
              callbacks=[TimeCallback(), csv_logger, early_stopping])
    write_results_to_plot(csv_file=csv_name,
                          destination_file=asset_name)
