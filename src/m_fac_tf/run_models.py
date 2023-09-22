# -*- coding: utf-8 -*-
import tensorflow as tf
from utils.utils import TrainingCallback
from utils.utils import ModelInspectionCallback
from optimizers.tester import TestOptimizer
from utils.utils import get_simple_raw_model


if __name__ == "__main__":
    (cifar_train_images, cifar_train_labels), (cifar_test_images, cifar_test_labels) = tf.keras.datasets.cifar10.load_data()
    cifar_train_images = cifar_train_images[:500]; cifar_train_labels = cifar_train_labels[:500]
    (mnist_training_data, mnist_training_labels), (mnist_test_data, mnist_test_labels) = tf.keras.datasets.mnist.load_data()
    mnist_training_data, mnist_test_data = mnist_training_data / 255, mnist_test_data / 255
    epochs = 5
    batch_size = 500
    optimizer = tf.keras.optimizers.Adam()
    optimizer2 = TestOptimizer(m=3, learning_rate=0.001)

    cifar_shape = (32,32,3)
    cifar_model = get_simple_raw_model(input_shape=cifar_shape, target_size=10) # 6 layers where 4 are trainable

    # cifar_model = tf.keras.applications.resnet50.ResNet50(include_top=False,
    #                                                 weights='imagenet',
    #                                                 input_shape=(32,32,3))
    

    #mnist_model = tf.keras.applications.resnet50.ResNet50(include_top=False,
    #                                                weights='imagenet',
    #                                                input_shape=(28,28))
    cifar_model.compile(optimizer=optimizer2,
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["accuracy"])
      
    #mnist_model.compile(optimizer=optimizer,
    #              loss = "sparse_categorical_crossentropy",
    #              metrics = ["accuracy"])
    cifar_model.summary()
    cifar_hist = cifar_model.fit(x=cifar_train_images,
                     y=cifar_train_labels,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     callbacks = [TrainingCallback()])
    
    # mnist_hist = mnist_model.fit(x=mnist_training_data,
    #                  y=mnist_training_labels,
    #                  batch_size=batch_size,
    #                  epochs=epochs,
    #                  verbose=1,
    #                  callbacks=[TrainingCallback()])    
