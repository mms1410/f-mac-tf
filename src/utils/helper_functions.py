"""Helper functions used in models and optimizers module."""
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import itertools
from omegaconf.listconfig import ListConfig
from pathlib import Path
import datetime
import platform
from typing import Union


def get_folder_dataframe(folder: Union[str, Path]) -> pd.DataFrame:
    """Aggregate logged metrics form multiple runs.

    This function create a single dataframe for logged data where a new column
    is created with filename.

    Args:
        folder: path to folder with csv files.

    Returns:
        pandas dataframe of logged data.
    """
    data = pd.DataFrame()
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            tmp = pd.read_csv(Path(folder, filename))
            tmp["filename"] = filename
            tmp["optimizer"] = os.path.basename(folder)
            tmp["experiment"] = os.path.basename(os.path.dirname(folder))
        data = pd.concat([data, tmp])
        # ToDo: add mean value

    return data


def write_os_info_to_file(output_file: Union[str, Path]) -> None:
    """

    Args:
        output_file:

    Returns:

    """
    with open(output_file, "w") as file:
        file.write("Operating System Information:\n")
        file.write(f"System: {platform.system()}\n")
        file.write(f"Node Name: {platform.node()}\n")
        file.write(f"Release: {platform.release()}\n")
        file.write(f"Version: {platform.version()}\n")
        file.write(f"Machine: {platform.machine()}\n")
        file.write(f"Processor: {platform.processor()}\n")



def set_log_filename_default(optimizer_name, modelname, batchsize, run):
    """

    Args:
        optimizer_name:
        modelname:
        batchsize:
        run:

    Returns:
        string with configuration information.
    """
    conf_name = datetime.datetime.now()
    conf_name = conf_name.strftime("%Y-%m-%d:%H:%mm")
    conf_name = conf_name + "_" + optimizer_name
    conf_name = conf_name + "_" + modelname
    conf_name = conf_name + "_batch-" + str(batchsize)
    conf_name = conf_name + "_run-" + str(run)

    return conf_name
def set_log_filename_mfac(optimizer_name, modelname, batchsize, run, m):
    """Set the filename for experiment configuration.

    Args:
        optimizer_name:
        batchsize:
        run:
        m:

    Returns:
        string of configuration information.
    """
    conf_name = datetime.datetime.now()
    conf_name = conf_name.strftime("%Y-%m-%d:%H:%mm")
    conf_name = conf_name + "_" + optimizer_name
    conf_name = conf_name + "_" + modelname
    conf_name = conf_name + "_batch-" + str(batchsize)
    conf_name = conf_name + "_m-" + str(m)
    conf_name = conf_name + "_run-" + str(run)

    return conf_name

def set_log_dir(root:str, name:str="logs") -> Path:
    """
    Args:
        root:

    Returns:
        string with path to log
    """
    log_dir_path = Path(root, name)
    if not log_dir_path.exists():
        log_dir_path.mkdir(parents=True)
    return log_dir_path

def residual_block(x: tf.keras.layers, filters: int):
    # Define a single residual block
    shortcut = x
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)

    return x


def build_resnet_20(input_shape, num_classes):
    # Define the input layer
    input_layer = tf.keras.layers.Input(shape=input_shape)

    # Initial convolution and max-pooling
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Stack residual blocks
    num_blocks = 6  # 6 residual blocks for a total of 20 layers
    for _ in range(num_blocks):
        x = residual_block(x, 16)

    # Global average pooling and final dense layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=input_layer, outputs=x, name='resnet20')

    return model

def build_resnet_32(input_shape, num_classes: int):
    """Build Resnet 32 model.

    Build a renet32 model based on the desired input shape and classes.

    Args:
        input_shape: Tuple of form (int, int, int).
        num_classes: integer number.

    Returns:
        keras model
    """
    # Define the input layer
    input_layer = tf.keras.layers.Input(shape=input_shape)

    # Initial convolution and max-pooling
    x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Stack residual blocks
    num_blocks = 10  # 10 residual blocks for a total of 32 layers
    for _ in range(num_blocks):
        x = residual_block(x, 16)

    # Global average pooling and final dense layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = tf.keras.Model(inputs=input_layer, outputs=x, name='resnet32')

    return model

def get_simple_raw_model(input_shape, target_size):
    """
    Create non-compilated model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=8, input_shape=input_shape, kernel_size=(4, 4),
                               activation='relu', name='conv_1'),
        tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', name='conv_2'),
        tf.keras.layers.Flatten(name='flatten'),
        tf.keras.layers.Dense(units=32, activation='relu', name='dense_1'),
        tf.keras.layers.Dense(units=target_size, activation='softmax', name='dense_2')])
    return model


class MatrixFifo:
    """
    Implements idea of fifo queue for tensorflow matrix.
    """
    def __init__(self, ncol):
        self.values = None
        self.ncol=ncol
        self.counter = 0
    
    def append(self, vector:tf.Tensor):
        """
        For k by m matrix and vecotr of dimension k by 1 move columns 2,...,m 'to left by one position' and substitute column m with vector.
        """
        if self.values is None:
            # first vector to append will determine nrow
            self.values = tf.Variable(tf.zeros(shape=[vector.shape[0], self.ncol]))
            self.values[:,-1].assign(tf.cast(vector, dtype=self.values.dtype))
        else:
            tmp = tf.identity(self.values)
            # update last column with new vector
            self.values[:,-1].assign(tf.cast(vector, dtype=self.values.dtype))
            # move other columns to left
            self.values[:,:-1].assign(tmp[:,1:])
        self.counter += 1

class RowWiseMatrixFifo:
    """Row-wise Matrix fifo queue.

    The top row contains the newest vector (row-wise).
    The matrix is initializes with zeros and when appended the firt m-1 rows
    move rown row down and the row on top is replaced by vector.
    """
    def __init__(self, m):
        self.values = None
        self.nrow = m
        self.counter = 0  # tf.Variable(0, dtype=tf.int32)
    
    def append(self, vector: tf.Tensor):
        """Append vector to fifoMatrix

        Append vector to first row and update all other rows, 
        where row i contains values of former row i-1.
        The first appended vector determines ncol.

        Args:
            vector: tf.Vector of gradients
        """
        if self.values is None:
            # init zero matrix
            # this is done here so the shape of vector determines ncol
            # and is not set at init.
            self.values = tf.Variable(tf.zeros(shape=[self.nrow, vector.shape[0]]))  # noqa E501
        
        # first m-1 rows are part of updated fifo matrix.
        maintained_values = tf.identity(self.values[:self.nrow - 1, :])
        # move row i is now former row i - 1.
        self.values[1:, :].assign(maintained_values)
        # update firt row with new vector.
        self.values[0, :].assign(vector)
        # increment counter
        self. counter += 1  # self.counter.assign_add(1)

    def reset(self):
        self.counter = 0
        self.values = None

def deflatten(flattened_grads: tf.Variable, shapes_grads: list[tf.shape]) -> tuple[tf.Variable]:  # noqa E501
    """Deflatten a tensorflow vector.
    
    Args:
        flattened_grads: flattened gradients.
        shape_grads: shape in which to reshape
    
    Return:
        tuple of tf.Variables
    """
    shapes_total = list(map(lambda x: tf.reduce_prod(x), shapes_grads))
    intermediate = tf.split(flattened_grads, shapes_total, axis=0)   # noqa E501
    deflattened = [tf.reshape(grad, shape) for grad, shape in zip(intermediate, shapes_grads)]  # noqa E501
    return deflattened


def write_results_to_plot(csv_file: str, destination_file: str) -> None:
    """
    """
    df = pd.read_csv(csv_file)
    # Extract data for each metric
    epochs = df["epoch"]
    accuracy = df["accuracy"]
    elapsed_time = df["elapsed_time"]
    loss = df["loss"]
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Metrics Over Epochs")
    # Plot accuracy
    axes[0, 0].plot(epochs, accuracy, label="accuracy", color="blue")
    axes[0, 0].set_title("accuracy")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("accuracy")
    # Plot elapsed_time
    axes[0, 1].plot(epochs, elapsed_time, label="elapsed_time", color="green")
    axes[0, 1].set_title("elapsed_time")
    axes[0, 1].set_xlabel("epochs")
    axes[0, 1].set_ylabel("elapsed_time")
    # Plot loss
    axes[1, 0].plot(epochs, loss, label="loss", color='red')
    axes[1, 0].set_title("loss")
    axes[1, 0].set_xlabel("epochs")
    axes[1, 0].set_ylabel("loss")
    # Adjust layout
    plt.tight_layout()
    # Save the figure as a PNG image
    plt.savefig(destination_file)


def get_param_combo_list(params:dict) -> List:
    """Create List of possible param combinations.

    Args:
        params: dictionary of possible param constellations.

    Returns:
        A list consisting of a dictioary for each combination.
    """
    # create all combinations
    keys = list(params.keys())
    # create a list with a list for each dimension in which cartesian product is computed.
    # if a value is already a list then just take value else list of value
    # if read from yaml via hydra element is not list but listConfig type.
    values = [value if isinstance(value, (ListConfig, list)) else [value] for value in params.values() ]

    combinations = itertools.product(*values)
    combinations = list(combinations)
    # append in list
    result_dicts = []
    for combo in combinations:
        result_dict = {key: val for key, val in zip(keys, combo)}
        result_dicts.append(result_dict)

    return result_dicts

if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]
    folder= Path(project_dir, "logs", "experiment1",  "SGD")
    data = get_folder_dataframe(folder)
    print(data.head())
