"""Set of models used for MNIST Kaggle comp."""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from utils import get_random_minibatches
from utils import load_training_set, one_hot, norm_imgs
from utils import split_train_test


def init_params(input_size, num_layers=3, num_nodes=256):
    params = {}

    previous_layer_size = input_size
    for layer in range(1, num_layers):
        weights = tf.get_variable('W{0}'.format(layer), [
            previous_layer_size, num_nodes],
            initializer=tf.contrib.layers.xavier_initializer(seed=1))
        bias = tf.get_variable(
            'b{0}'.format(layer), [1, num_nodes],
            initializer=tf.zeros_initializer())
        params['W{0}'.format(layer)] = weights
        params['b{0}'.format(layer)] = bias
        previous_layer_size = num_nodes

    out = tf.get_variable(
        'out', [previous_layer_size, 10],
        initializer=tf.contrib.layers.xavier_initializer(seed=1))
    bout = tf.get_variable('bout', [1, 10], initializer=tf.zeros_initializer())

    params.update({'out': out, 'bout': bout})

    return params


def forward_prop(X, params, num_layers):
    previous_layer = X

    for layer in range(1, num_layers):
        # Hidden fully connected layer with 256 neurons
        Z = tf.matmul(
            previous_layer,
            params['W{0}'.format(layer)]) + params['b{0}'.format(layer)]
        previous_layer = tf.nn.relu(Z)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.add(
        tf.matmul(previous_layer, params['out']), params['bout'])

    return out_layer


class SimpleNN(object):

    """Simple NN with a customisable number of layers."""

    def __init__(
        self, input_size, output_size, num_layers=3,
        nodes_per_layer=256, learning_rate=0.001
    ):
        """Init layer size."""
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.nodes_per_layers = nodes_per_layer
        self.learning_rate = learning_rate
        self.params = init_params(input_size, num_layers, nodes_per_layer)

    def predict(self, x):
        """Perform forward prop."""
        return forward_prop(x, self.params, num_layers=self.num_layers)

    def loss(self, logits, batch_y):
        self.cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=batch_y, name='loss'))
        return self.cost

    def optimize(self, batch_x, batch_y):
        return tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.cost)


def train_model(sess, model, X_train, Y_train, X_dev, Y_dev, mini_batch_size=64, num_epochs=50, print_every=10, model_path=None):
    X = tf.placeholder(tf.float32, shape=(None, model.input_size), name='X')
    Y = tf.placeholder(tf.float32, shape=(None, model.output_size), name='Y')

    costs = []

    logits = model.predict(X)
    cost = model.loss(logits, Y)
    optimizer = model.optimize(X, Y)

    tf.logging.set_verbosity('DEBUG')

    saver = tf.train.Saver()

    if model_path:
        saver.restore(sess, model_path)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    for epoch in range(1, num_epochs + 1):
        epoch_cost = 0.0
        num_mini_batches = 0
        train_batches = get_random_minibatches(X_train, Y_train, mini_batch_size=mini_batch_size)
        for X_mini, Y_mini in train_batches:
            num_mini_batches += 1
            _, batch_cost = sess.run([optimizer, cost], feed_dict={X: X_mini, Y: Y_mini})
            epoch_cost += batch_cost

        epoch_cost /= num_mini_batches

        if epoch % print_every == 0:
            print("Cost at {0} epochs is {1}".format(epoch, epoch_cost))
        if epoch % 5 == 0:
            costs.append(epoch_cost)

    if model_path:
        saver.save(sess, model_path)

    params = sess.run(model.params)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}, session=sess))
    print ("Test Accuracy:", accuracy.eval({X: X_dev, Y: Y_dev}, session=sess))

    return params, costs


if __name__ == '__main__':
    imgs, labels = load_training_set('./data/train.csv')
    labels = one_hot(labels, 10)
    imgs = norm_imgs(imgs)

    X_train, Y_train, X_dev, Y_dev = split_train_test(imgs, labels, 0.2)

    # Train the model.
    sess = tf.Session()
    params, costs = train_model(
        sess, model=SimpleNN(input_size=X_train.shape[1], output_size=Y_train.shape[1]),
        X_train=X_train, Y_train=Y_train, X_dev=X_dev, Y_dev=Y_dev,
        mini_batch_size=64, num_epochs=1, model_path='mnist_simple_nn')
    sess.close()
