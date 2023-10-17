import tensorflow as tf
from tensorflow.keras.applications import ResNet50
#logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
#tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
#                                                 histogram_freq = 1,
#                                                profile_batch = '500,520')

if __name__ == "__main__":
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    print(base_model)