# Adapted from https://danijar.com/structuring-your-tensorflow-models/
# Using a different architecture, though

import coco
import functools
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from PIL import Image
from cache import cache
from enum import Enum

import tensorflow as tf
from tensorflow.python.keras.applications import VGG16


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without parenthesis if no 
    arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped function will only be 
    executed once. Subsequent calls to it will directly return the result so that operations are 
    added to the graph only once.

    The operations added by the function live within a tf.variable_scope(). If this decorator is 
    used with arguments, they will be forwarded to the variable scope. The scope name defaults to 
    the name of the wrapped function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Dataset(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class Model:

    def __init__(self, image, label, mode):
        self.image = image
        self.label = label
        self.mode = mode
        self.prediction
        self.optimize
        self.error

    @define_scope(initializer=tf.contrib.layers.xavier_initializer())
    def prediction(self):
        x = tf.reshape(self.image, [-1, 28, 28, 1])
        x = tf.layers.conv2d(
            inputs=x, filters=32, kernel_size=[5, 5], padding="SAME", activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
        x = tf.layers.conv2d(
            inputs=x, filters=64, kernel_size=[5, 5], padding="SAME", activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(inputs=x, pool_size=[2, 2], strides=2)
        x = tf.reshape(x, [-1, 7 * 7 * 64])
        x = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
        x = tf.layers.dropout(inputs=x, rate=0.4, training=self.mode == "train")

        x = tf.layers.dense(x, units=10)
        return x

    @define_scope
    def optimize(self):
        loss = tf.losses.softmax_cross_entropy(self.label, logits=self.prediction)
        # Add loss to TensorBoard summary
        tf.summary.scalar("Softmax loss", loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
        return optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    @define_scope
    def error(self):
        mistakes = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        err_mean = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        tf.summary.scalar("Error", err_mean)
        return err_mean


def load_image(path, size=None):
    """
    Load the image from the given file path and resize it to the given size if not None.
    """

    # Load the image using PIL
    img = Image.open(path)

    # Resize image if desired
    if not size is None:
        img = img.resize(size=size, resample=Image.LANCZOS)

    # Convert image to numpy array
    img = np.array(img)

    # Scale image pixels so they fall between 0 and 1
    img = img / 255.0

    # Convert 2-dim grayscale array to 3-dim RGB array (1 channel to 3)
    if (len(img.shape) == 2):
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    return img


def show_image(idx, filenames_set, captions_set):
    """
    Load and plot an image from the training or validation set with the given index
    """

    # Use an image from the training set
    dir = coco.train_dir
    filename = filenames_set[idx]
    captions = captions_set[idx]

    # Path for the image file
    path = os.path.join(dir, filename)

    # Print the captions for the image
    for caption in captions:
        print(caption)

    # Load the image and plot it
    img = load_image(path)
    plt.imshow(img)
    plt.show()


def main():
    # Logging stuff
    now = datetime.now()
    logdir = "/tmp/image_captioning/" + now.strftime("%Y%m%d-%H%M%S") + "/"

    # Get Coco dataset
    coco.set_data_dir("../Datasets/coco/")
    coco.maybe_download_and_extract()

    # Get file names and captions
    _, filenames_train, captions_train = coco.load_records(train=True)

    num_images_train = len(filenames_train)
    print("Number of training images = {}".format(num_images_train))

    image_model = VGG16(include_top=True, weights='imagenet')
    image_model.summary()


if __name__ == '__main__':
    main()