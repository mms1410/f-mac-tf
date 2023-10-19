import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
from pathlib import Path
from src.utils.helper_functions import get_param_combo_list
from src.utils.datasets import load_optimizer
from src.utils.datasets import get_dataset, get_model
from src.utils.monitor_training import CustomCSVLogger
from numpy import unique as unique
from src.utils.helper_functions import set_log_dir, set_log_filename_mfac, set_log_filename_default

project_dir = Path(__file__).resolve().parents[1]
config_path = Path(project_dir, "conf")


@hydra.main(config_path=str(config_path), config_name="experiment1")
def main(conf: DictConfig):
    log_dir = set_log_dir(root=project_dir)
    config_name = "experiment1"
    log_dir = set_log_dir(root=log_dir, name=config_name)
    dataset_name = conf.dataset
    model_name = conf.model
    x_train, y_train, x_test, y_test = get_dataset(dataset_name)
    input_shape = x_train.shape[1:]
    n_classes = len(unique(y_train))
    model = get_model(model_name,
                      n_classes=n_classes,
                      input_shape=input_shape)
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

                    # set variables for configuration logging
                    if "mfac" in optimizer_name.lower():
                        conf_name = set_log_filename_mfac(optimizer_name,
                                                          model_name,
                                                          batch_size,
                                                          run,
                                                          param_combo.get("m"))
                    else:
                        conf_name = set_log_filename_default(optimizer_name,
                                                             model_name,
                                                             batch_size,
                                                             run)
                    # for each optimizer create dedicated log folder
                    optimizer_log_dir = Path(project_dir, log_dir, optimizer_name)

                    if not optimizer_log_dir.is_dir():
                        optimizer_log_dir.mkdir(exist_ok=True)

                    csv_logger = CustomCSVLogger(Path(optimizer_log_dir, conf_name + ".csv"))

                    model.compile(optimizer=optimizer,
                                  loss=loss,
                                  metrics=["accuracy"],
                                  run_eagerly=True)

                    model.fit(x_train, y_train,
                              validation_data=(x_test, y_test),
                              epochs=epochs,
                              batch_size=batch_size,
                              callbacks=[csv_logger])

if __name__ == "__main__":
    conf = OmegaConf.load(Path(config_path, "experiment1.yaml"))
    main()