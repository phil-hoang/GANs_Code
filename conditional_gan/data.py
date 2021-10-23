# Returns a tensorflow dataset.
# Should contain MNIST and CIFAR 10
from tensorflow import keras
import tensorflow as tf
import numpy as np


def mnist(batch_size):
    # We'll use all the available examples from both the training and test
    # sets.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    all_digits = np.concatenate([x_train, x_test])
    all_labels = np.concatenate([y_train, y_test])

    image_size = 28
    num_channels = 1
    
    #image_size = x_train.shape[1]
    #if len(x_train[0]) == 2:
    #    num_channels = 1
    #else:
    #   num_channels = 3
    

    # Scale the pixel values to [0, 1] range, add a channel dimension to
    # the images, and one-hot encode the labels.
    all_digits = all_digits.astype("float32") / 255.0
    all_digits = np.reshape(all_digits, (-1, image_size, image_size, num_channels))
    all_labels = keras.utils.to_categorical(all_labels, 10)
    

    # Create tf.data.Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((all_digits, all_labels))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Extract infos
    num_classes = all_labels.shape[1]
    
    print(f"Shape of training images: {all_digits.shape}")
    print(f"Shape of training labels: {all_labels.shape}")

    return dataset, image_size, num_channels, num_classes