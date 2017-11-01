import math
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import timeit

def run(x_test):

    mnist = read_data_sets("/tmp/data", one_hot=True)
    Xtrain = mnist.train.images
    ytrain = mnist.train.labels

    # Set k value
    k = 3

    # Initialize placeholders
    X = tf.placeholder("float", [None, 784])
    y = tf.placeholder("float", [784])

    # Initalize distance
    dist = tf.reduce_sum(tf.abs(tf.subtract(X, y)), axis=1)

    # Initalize predicted class
    yp = tf.argmin(dist, 0)
    init = tf.global_variables_initializer()

    # Initalize array of predicted classes
    predicted_y_test = []

    with tf.Session() as sess:
        sess.run(init)
        for i in range(len(x_test)):
        	# Get the predicted class
            predicted_class = sess.run(yp, feed_dict={X: Xtrain, y: x_test[i, :]})

            # Append predicted class to predicted test array
            predicted_y_test.append(np.argmax(ytrain[predicted_class]))

    return predicted_y_test


def hyperparameters_search():
    raise NotImplementedError

if __name__ == '__main__':
    hyperparameters_search()
