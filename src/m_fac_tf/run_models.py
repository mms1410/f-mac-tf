# -*- coding: utf-8 -*-
import tensorflow as tf
from utils.utils import TrainingCallback
from utils.utils import ModelInspectionCallback
from optimizers.m_fac import MFAC


if __name__ == "__main__":
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images = train_images[:500]; train_labels = train_labels[:500]
    epochs = 5
    batch_size = 128
    model = tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                    weights='imagenet',
                                                    input_shape=(32,32,3))
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer,
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["accuracy"])
    hist = model.fit(x=train_images,
                     y=train_labels,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     callbacks=[ModelInspectionCallback()])