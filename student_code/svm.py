import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.examples.tutorials.mnist import input_data
import timeit


def run(x_test):
    # TODO: Write your algorithm here
    
    mnist = input_data.read_data_sets("/tmp/data",one_hot=False)

    ntrain = mnist.train.num_examples
    Xtrain = mnist.train.images
    ytrain = mnist.train.labels
    
    ntest = mnist.test.num_examples
    Xtest = mnist.test.images
    ytest = mnist.test.labels
    
    nvalidation = mnist.validation.num_examples
    Xvalidation = mnist.validation.images
    yvalidation = mnist.validation.labels

    n_inputs = 28 * 28
    batch_size = 200

    X = tf.placeholder(tf.float32,[None, n_inputs])
    y = tf.placeholder(tf.float32,[batch_size, 1])
    
    lr = 0.0001 # learning rate
    lam_val = 1 # regularization parameter

    def get_training_op(score, W):
        # mean squared error as loss function
        l2_loss = 0.5 * lam_val * tf.reduce_mean(tf.square(W))
        hinge_loss = lam_val * tf.reduce_sum(tf.maximum(tf.zeros([batch_size,10]), 1 - y*score))
        loss = l2_loss + hinge_loss
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr) # another better optimizer
        training_op = optimizer.minimize(loss)
        return training_op

    weights = []
    biases = []
    scores = []
    training_ops = []

    for i in range(10):
        W = tf.Variable(tf.random_normal([n_inputs, 1]))
        b = tf.Variable(tf.zeros([1]))
        score = tf.matmul(X,W) + b
        training_op = get_training_op(score, W)

        weights.append(W)
        biases.append(b)
        scores.append(score) 
        training_ops.append(training_op)   
    
    init = tf.global_variables_initializer()

    n_epochs = 50
    y_predicted_test = []
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            # compute model
            n_batches = mnist.train.num_examples // batch_size
            for iteration in range(n_batches):
                X_batch,y_batch = mnist.train.next_batch(batch_size)

                for i in range(10):
                    y_vector = np.array([1 if y==i else -1 for y in y_batch])

                    y_vector = y_vector.reshape(batch_size, 1)
                    sess.run(training_ops[i], feed_dict={X:X_batch, y:y_vector})
                    weights[i].eval()
                    biases[i].eval()

        
        # Now that the model is trained, it is the test time!

        yp_tests = []
        for i in range(10):
            yp = scores[i].eval(feed_dict={X:x_test, W:weights[i].eval(), b:biases[i].eval()})
            yp_tests.append(yp)
        
        for i in range(len(x_test)):
            predicted_class = 0
            max_score = yp_tests[0][i]
            for index, test in enumerate(yp_tests):
                if test[i] > max_score:
                    max_score = test[i]
                    predicted_class = index
            y_predicted_test.append(predicted_class)

        return y_predicted_test


def hyperparameters_search():
    raise NotImplementedError


if __name__ == '__main__':
    hyperparameters_search()
