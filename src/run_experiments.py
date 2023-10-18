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
from numpy import unique as unique
from tensorflow.keras.callbacks import TensorBoard as TensorBoard
from src.utils.helper_functions import set_log_dir


project_dir = Path(__file__).resolve().parents[1]
config_path = Path(project_dir, "conf")



@hydra.main(config_path=str(config_path), config_name="experiment1")
def main(conf: DictConfig):
    log_dir = set_log_dir(root=project_dir)
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

                    conf_name = datetime.datetime.now()
                    conf_name = conf_name.strftime("%Y-%m-%d:%H")
                    conf_name = (conf_name + "_" +
                                 optimizer_name +
                                          "_" + model_name + "_"
                                          "_batchsize" +
                                 str(batch_size) +
                                          "_run" +
                                 str(run))
                    print(conf_name)
                    model.compile(optimizer=optimizer,
                                  loss=loss,
                                  metrics=["accuracy"],
                                  run_eagerly=True)

                    csv_logger = CSVLogger(Path(log_dir, conf_name + ".csv"),
                                           separator=",",
                                           append=False)
                    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

                    model.fit(x_train, y_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              callbacks=[tensorboard_callback])

if __name__ == "__main__":
    conf = OmegaConf.load(Path(config_path, "experiment1.yaml"))
    main()