from datetime import datetime
import tensorflow as tf
from src.m_fac_tf.utils.monitor_training import ModelInspectionCallback
from optimizers.tester import TestOptimizer
from src.m_fac_tf.utils.helper_functions import get_simple_raw_model


if __name__ == "__main__":
    (cifar_train_images, cifar_train_labels), (cifar_test_images, cifar_test_labels) = tf.keras.datasets.cifar10.load_data()
    cifar_train_images = cifar_train_images[:500]; cifar_train_labels = cifar_train_labels[:500]
    (mnist_training_data, mnist_training_labels), (mnist_test_data, mnist_test_labels) = tf.keras.datasets.mnist.load_data()
    mnist_training_data, mnist_test_data = mnist_training_data / 255, mnist_test_data / 255
    epochs = 10
    batch_size = 128
    optimizer = tf.keras.optimizers.Adam()
    optimizer2 = TestOptimizer(m=6, learning_rate=0.001)


    logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')

    ############## CIFAR
    #cifar_model = tf.keras.applications.resnet50.ResNet50(include_top=False,
    #                                                 weights='imagenet',
    #                                                 input_shape=(32,32,3))

    cifar_shape = (32,32,3)
    cifar_model = get_simple_raw_model(input_shape=cifar_shape, target_size=10) # 6 layers where 4 are trainable

    cifar_model.compile(optimizer=optimizer,
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["accuracy"])
    
    cifar_hist = cifar_model.fit(x=cifar_train_images,
                     y=cifar_train_labels,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                    callbacks = [tboard_callback])



    ############## MNIST
    #mnist_model = tf.keras.applications.resnet50.ResNet50(include_top=False,
    #                                                weights='imagenet',
    #                                                input_shape=(32,32, 3))
    #mnist_model.compile(optimizer=optimizer,
    #              loss = "sparse_categorical_crossentropy",
    #              metrics = ["accuracy"])
    #mnist_hist = mnist_model.fit(x=mnist_training_data,
    #                  y=mnist_training_labels,
    #                  batch_size=batch_size,
    #                  epochs=epochs,
    #                  verbose=1)    