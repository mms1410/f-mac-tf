import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
from pathlib import Path
from src.utils.helper_functions import get_param_combo_list
from src.utils.datasets import load_optimizer
from src.utils.datasets import get_dataset, get_model
import datetime
from tensorflow.keras.callbacks import CSVLogger
from src.utils.monitor_training import TimeCallback

project_dir = Path(__file__).resolve().parents[1]
config_path = Path(project_dir, "conf")


@hydra.main(config_path=str(config_path), config_name="experiment1")
def main(conf: DictConfig):
    dataset_name = conf.dataset
    model_name = conf.model
    x_train, y_train, x_test, y_test = get_dataset(dataset_name)
    if dataset_name == "cifar10":
        input_shape = (32, 32, 3)
    elif dataset_name == "mnist":
        input_shape = (28, 28, 1)
    model = get_model(model_name, n_classes=10, input_shape=input_shape)
    epochs = conf.epochs
    loss = conf.loss
    for optimizer_name in conf.optimizer:
        optimizer_dict = conf["optimizer"][optimizer_name]
        optimizer_params = optimizer_dict.params
        param_combos = get_param_combo_list(optimizer_params)
        batch_size = optimizer_dict.batch_size
        for run in range(conf.runs):
                for param_combo in param_combos:
                    # set optimizer
                    if optimizer_name.lower().startswith("mfac"):
                        optimizer = load_optimizer(optimizer_name, param_combo)
                    elif optimizer_name.lower().startswith("f-mfac"):
                        optimizer = load_optimizer(optimizer_name, param_combo)
                    elif optimizer_name.lower() == "adam":
                        optimizer = load_optimizer(optimizer_name, param_combo)
                    elif optimizer_name.lower() == "sgd":
                        optimizer = load_optimizer(optimizer_name, param_combo)

                    configuration_name = datetime.datetime.now()
                    configuration_name = configuration_name.strftime("%Y-%m-%d:%H")
                    configuration_name = (configuration_name + "_" +
                                          optimizer_name +
                                          "_" + model_name + "_"
                                          "_batchsize" +
                                          str(batch_size) +
                                          "_run" +
                                          str(run))
                    print(configuration_name)
                    model.compile(optimizer=optimizer,
                                  loss=loss,
                                  metrics=["accuracy"],
                                  run_eagerly=True)

                    csv_logger = CSVLogger(configuration_name, separator=",", append=False)

                    model.fit(x_train, y_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              callbacks=[TimeCallback(), csv_logger])

if __name__ == "__main__":
    conf = OmegaConf.load(Path(config_path, "experiment1.yaml"))
    main()