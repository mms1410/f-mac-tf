"""Helper functions used in models and optimizers module."""
from typing import List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import itertools
from omegaconf.listconfig import ListConfig


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

def _setupMatrices(self, G:tf.Variable, D:tf.Variable, B:tf.Variable) -> None:
        """
        """
        # init matrices
        self.G = self.grad_fifo.values
        self.D = tf.Variable(tf.math.scalar_mul(self.damp, tf.matmul(self.G, tf.transpose(self.G))))
        self.B = tf.Variable(tf.math.scalar_mul(self.damp, tf.Variable(tf.linalg.eye(self.m))))
        # Compute D
        for idx in range(1, self.m):
            denominator = tf.math.pow((self.m + self.D[idx - 1, idx - 1]), -1)
            test = self.D[idx:, idx:] - denominator * tf.matmul(tf.transpose(self.D[idx - 1:, idx:]), self.D[idx - 1:, idx:])
            self.D[idx:, idx:].assign(test)
        self.D = tf.linalg.band_part(self.D, 0, -1)
        # Compute B
        for idx in range(1, self.m):
            denominator = self.m + tf.linalg.diag_part(self.D)[:idx]
            tmp = tf.math.divide(-self.D[:idx, idx], denominator)
            tmp = tf.transpose(tmp)
            to_assign = tf.linalg.matvec(self.B[:idx, :idx], tmp)
            self.B[idx, :idx].assign(to_assign)

def _compute_InvMatVec(self, x):
    """
    Compute \hat{F_{m}}\bm{x} for precomputed D and B
    """
    self.G = self.grad_fifo.values
    q = tf.Variable(tf.linalg.matvec(self.G, x))
    q0 = q[0] / (self.m + self.D[0, 0])
    q[0].assign(q0)
    for idx in range(1, self.m):
        tmp = q[idx:] - tf.math.scalar_mul(q[idx - 1], tf.transpose(self.D[idx - 1, idx:]))
        q[idx:].assign(tmp)
    denominator =self.m + tf.linalg.diag_part(self.D)
    q = q / denominator
    tmp = tf.linalg.matvec(self.B, q)
    a = tf.transpose(tf.linalg.matvec(self.G, tmp, transpose_a=True))
    b = tf.math.scalar_mul(self.damp, x)
    result = a - b
    return result


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
    params = {'m': [50, 100, 300, 500], 'damp': 1e-07, 'learning_rate': 0.001}
    param_dicts = get_param_combo_list(params)
