# -*- coding: utf-8 -*-
"""Download datasets for benchmarking in dedicated data folder"""

import os
from os.path import dirname as odir
import tensorflow as tf

PROJECT_ROOT = __file__
PROJECT_ROOT = odir(odir(odir(odir(PROJECT_ROOT))))

data_folder = os.path.join(PROJECT_ROOT, "data")


def check_data_folder(path_folder: str = data_folder) -> bool:
    """ Check existence of data folder"""
    if os.path.isdir(path_folder):
        return True
    else:
        return False


def check_cifar_10(path_folder: str = data_folder) -> bool:
    """ Check whether CIFAR_10 already in data folder"""


if __name__ == "__main__":
    print("datadir: ", data_folder)
    tf.keras.datasets.cifar10.load_data()
