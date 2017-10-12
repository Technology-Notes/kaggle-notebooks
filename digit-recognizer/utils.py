import csv
import math

import numpy as np
import tensorflow as tf


def get_random_minibatches(X, Y, mini_batch_size=64, seed=0):
    """Return a list of minibatches in format ([x...], [y...])."""
    np.random.seed(seed)
    m = X.shape[0]

    # Create a random list of numbers
    permutations = list(np.random.permutation(m))
    shuffled_X = X[permutations, :]
    shuffled_Y = Y[permutations, :]

    num_mini_batches = math.floor(m / mini_batch_size)

    for i in range(num_mini_batches):
        start_pos = i * mini_batch_size
        end_pos = start_pos + mini_batch_size

        mini_batch_X = shuffled_X[start_pos:end_pos, :]
        mini_batch_Y = shuffled_Y[start_pos:end_pos, :]

        yield (mini_batch_X, mini_batch_Y)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[m - mini_batch_size:, :]
        mini_batch_Y = shuffled_Y[m - mini_batch_size:, :]
        yield (mini_batch_X, mini_batch_Y)


def load_training_set(filepath, max_rows=None):
    fh = open(filepath)
    reader = csv.reader(fh)

    # Skip the headers
    next(reader)

    labels = []
    imgs = []
    for count, line in enumerate(reader):
        label, img = line[0], line[1:]
        labels.append(label)
        imgs.append(img)

        if max_rows and count > max_rows:
            break

    return np.uint8(imgs), np.uint8(labels)


def one_hot(arr, num_labels):
    C = tf.constant(num_labels, name='c')
    one_hot_matrix = tf.one_hot(arr, C, axis=1, dtype=tf.uint8)

    sess = tf.Session()
    labels_one_hot = sess.run(one_hot_matrix)
    sess.close()

    return labels_one_hot


def norm_imgs(arr):
    return arr / 255


def split_train_test(X, Y, ratio):
    dev_set_size = int(X.shape[0] * ratio)

    train_X = X[dev_set_size:, :]
    train_Y = Y[dev_set_size:, :]

    dev_X = X[:dev_set_size, :]
    dev_Y = Y[:dev_set_size, :]

    return train_X, train_Y, dev_X, dev_Y
