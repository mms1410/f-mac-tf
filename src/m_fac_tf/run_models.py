# -*- coding: utf-8 -*-
import tensorflow as tf
from utils.utils import TrainingCallback
from utils.utils import ModelInspectionCallback
from optimizers.tester import TestOptimizer


if __name__ == "__main__":
    (cifar_train_images, cifar_train_labels), (cifar_test_images, cifar_test_labels) = tf.keras.datasets.cifar10.load_data()
    cifar_train_images = cifar_train_images[:500]; cifar_train_labels = cifar_train_labels[:500]
    
    epochs = 5
    batch_size = 500
    optimizer = tf.keras.optimizers.Adam()
    optimizer2 = TestOptimizer(m=3, learning_rate=0.001)

    cifar_model = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                    weights='imagenet',
                                                    input_shape=(32,32,3))
    cifar_model.compile(optimizer=optimizer2,
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["accuracy"])
    cifar_hist = cifar_model.fit(x=cifar_train_images,
                     y=cifar_train_labels,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     callbacks = [TrainingCallback()])