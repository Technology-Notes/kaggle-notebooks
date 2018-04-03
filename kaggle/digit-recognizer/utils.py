import csv
import math

from io import BytesIO
import PIL.Image
import IPython.display
import matplotlib.pyplot as plt
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


def show_digit(a, fmt='png'):
    a = a.reshape((28, 28))
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))


def plot_costs(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per 5)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def predict(x, weights, biases, forward_prop):
    weights = {key: tf.convert_to_tensor(var) for key, var in weights.items()}
    biases = {key: tf.convert_to_tensor(var) for key, var in biases.items()}

    x_ = tf.placeholder(tf.float32, [None, x.shape[1]])

    out_layer = forward_prop(x_, weights, biases)
    p = tf.argmax(out_layer, 1)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x_: x})
    sess.close()
    return prediction


def load_test_set(filepath):
    fh = open('./data/test.csv')
    test_reader = csv.reader(fh)

    # Skip header line
    next(test_reader)

    test_imgs = [line for line in test_reader]

    return np.uint8(test_imgs)


def save_predictions(predictions, filepath='output.csv'):
    """Save predictions to an output path."""
    output = open(filepath, 'w')
    output.write(u'ImageId,Label\n')
    [output.write(u'{0},{1}\n'.format(count + 1, pred))
        for count, pred in enumerate(predictions)]
    output.close()


class Epochs(object):

    def __init__(
        self, sess, optimizer, cost, accuracy, x, y, num_epochs=100,
        mini_batch_size=64, print_every=10
    ):
        self.sess = sess
        self.optimizer = optimizer
        self.cost = cost
        self.accuracy = accuracy
        self.x = x
        self.y = y
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.print_every = print_every

    def run(self, train_x, train_y):
        num_mini_batches = train_x.shape[0] / self.mini_batch_size

        costs = []

        for epoch in range(1, self.num_epochs + 1):
            epoch_cost = 0.
            for mini_x, mini_y in get_random_minibatches(train_x, train_y, self.mini_batch_size):
                _, batch_cost = self.sess.run([self.optimizer, self.cost], feed_dict={self.x: mini_x, self.y: mini_y})
                epoch_cost += batch_cost / num_mini_batches

            if epoch % self.print_every == 0:
                print("Cost at {0} epochs = {1}.".format(epoch, epoch_cost))
            if epoch % 5 == 0:
                costs.append(epoch_cost)

        return costs
