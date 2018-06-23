# Adapted from https://danijar.com/structuring-your-tensorflow-models/
# Using a different architecture, though

import functools
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.utils import shuffle
from datetime import datetime


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


def main():
    # Logging stuff
    now = datetime.now()
    logdir = "/tmp/mnist_structured/" + now.strftime("%Y%m%d-%H%M%S") + "/"

    # Get MNIST data
    mnist = input_data.read_data_sets('../Datasets/mnist/', one_hot=True)
    train_images_shuf, train_labels_shuf = shuffle(mnist.train.images, mnist.train.labels)

    # Split into training and validation sets
    train_size = train_images_shuf.shape[0]
    split_size = int(train_size * 0.9)

    train_images = train_images_shuf[:split_size]
    train_labels = train_labels_shuf[:split_size]
    val_images = train_images_shuf[split_size:]
    val_labels = train_labels_shuf[split_size:]

    NUM_EPOCHS = 20  # How many epochs should we train for?
    BATCH_SIZE = 64  # How big are our minibatches?
    NUM_BATCHES = int(split_size / BATCH_SIZE)
    VAL_BATCHES = int((train_size - split_size) / BATCH_SIZE)
    # Placeholder to switch between batch sizes
    batch_size = tf.placeholder(tf.int64)

    image = tf.placeholder(tf.float32, [None, 784])
    label = tf.placeholder(tf.float32, [None, 10])
    mode = tf.placeholder(tf.string)

    # datasets from numpy arrays
    dataset = tf.data.Dataset.from_tensor_slices((image, label)).batch(batch_size).repeat()
    iter = dataset.make_initializable_iterator()
    features, labels = iter.get_next()

    model = Model(features, labels, mode)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
        summary_op = tf.summary.merge_all()
        counter = 0
        sess.run(tf.global_variables_initializer())
        print('Training!')
        for i in range(NUM_EPOCHS):
            # Initialize iterator with training data
            sess.run(
                iter.initializer,
                feed_dict={
                    image: train_images,
                    label: train_labels,
                    batch_size: BATCH_SIZE
                })
            for _ in range(NUM_BATCHES):
                summary, _ = sess.run([summary_op, model.optimize], feed_dict={mode: "train"})
                writer.add_summary(summary, global_step=counter)
                counter += 1
                #sess.run(model.optimize, feed_dict={mode: "train"})

            # Iterator with validation data
            sess.run(
                iter.initializer,
                feed_dict={
                    image: val_images,
                    label: val_labels,
                    batch_size: val_images.shape[0]
                })
            # Iterate over validation set
            total_error = 0.0
            for _ in range(VAL_BATCHES):
                total_error += sess.run(model.error, feed_dict={mode: "eval"})
            mean_error = total_error / VAL_BATCHES
            print('Epoch {}: Validation error {:6.2f}%'.format(i, 100 * mean_error))

        writer.close()


if __name__ == '__main__':
    main()