from datetime import datetime as datetime
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import CSVLogger
from src.m_fac_tf.utils.monitor_training import ModelInspectionCallback
from src.m_fac_tf.optimizers.MFAC import Mfac
from src.m_fac_tf.utils.datasets import get_model_and_dataset


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[3]
    epochs = 5
    m = 20
    damp = 1e-8
    batch_size = 384
    optimizer = Mfac(m=m, damp=damp)
    dataset_name = "resnet50"  # resnet50, resnet101
    model, x_train, y_train, x_test, y_test = get_model_and_dataset(dataset_name)  # noqa E501

    current_datetime = datetime.now()
    iso_format = current_datetime.strftime("%Y-%m-%dT%H:%00:00")
    log_name = f"experiment_{iso_format}_{optimizer.name}_{dataset_name}"
    if optimizer.name == "MFAC":
        log_name = log_name + f"_m{m}_damp{damp:e}"
    log_name = Path(project_dir, "logs", log_name + ".log")
    csv_logger = CSVLogger(log_name)
    model.compile(optimizer=optimizer,
                  loss="categorical_crossentropy",
                  metrics=["accuracy"],
                  run_eagerly=True)
    model.fit(x_train, y_train, epochs=epochs,
              batch_size=batch_size,
              callbacks=[ModelInspectionCallback(), csv_logger])
