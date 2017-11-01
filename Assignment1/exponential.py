import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import timeit


def run(x_test):
    # TODO: Write your algorithm here
    
    mnist = read_data_sets("/tmp/data",one_hot=False)

    ntrain = mnist.train.num_examples
    Xtrain = mnist.train.images
    ytrain = mnist.train.labels

    n_inputs = 28 * 28
    batch_size = 200
    n_classes = 10

    # Initialize X and Y matrices
    X = tf.placeholder(tf.float32,[None, n_inputs])
    y = tf.placeholder(tf.float32,[batch_size, 1])
    
    lr = 0.01 # learning rate
    lam_val = 0.00001 # regularization parameter
    C = 0.001

    # Get the training op for each class score and W
    def get_training_op(score, W):
        # L2 regularlization loss
        l2_loss = 0.5 * lam_val * tf.reduce_mean(tf.square(W))

        # Get exponential loss
        exp_loss = C * tf.reduce_sum(tf.exp(-y*score))

        # Get binary classifier
        loss = l2_loss + exp_loss

        # Get the optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr) # another better optimizer

        # Get the training op
        training_op = optimizer.minimize(loss)
        return training_op

    # Store all 10 weight matrices, biases, scores, and training ops
    weights = []
    biases = []
    scores = []
    training_ops = []

    # Create 10 training ops for each classes by getting W, b, and score for each class
    for i in range(n_classes):
        #W = tf.Variable(tf.random_normal([n_inputs, 1]))
        #b = tf.Variable(tf.zeros([1]))

        # Initalize W parameter with all the same values
        W = tf.Variable(tf.fill([n_inputs, 1], 0.0000001))

        # Initialize b parameter
        b = tf.Variable(tf.fill([1], 0.01))

        # Get the score (yp)
        score = tf.matmul(X,W) + b

        # Get the training op using score and W
        training_op = get_training_op(score, W)

        # Store the weights, biases, scores, and training ops
        weights.append(W)
        biases.append(b)
        scores.append(score) 
        training_ops.append(training_op)   
    
    # Initalize all variables
    init = tf.global_variables_initializer()

    n_epochs = 50

    # Array of predicted values (what we will be returning to main)
    y_predicted_test = []
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            # Get the number of batches
            n_batches = mnist.train.num_examples // batch_size
            for iteration in range(n_batches):
                # Get training data
                X_batch,y_batch = mnist.train.next_batch(batch_size)

                # Initalize 10 y vectors and convert y batch values to 1 and -1, where
                # the current class will have a value of 1 and all other classes -1
                for i in range(n_classes):
                    # Convert to 1,-1 for one-vs-all scheme
                    y_vector = np.array([1 if y==i else -1 for y in y_batch])
                    y_vector = y_vector.reshape(batch_size, 1)

                    # Run the training op for the specific class
                    sess.run(training_ops[i], feed_dict={X:X_batch, y:y_vector})
                    weights[i].eval()
                    biases[i].eval()

        
        # Run all classifiers and get the predicted tests of all classes
        yp_tests = []
        for i in range(n_classes):
            yp = scores[i].eval(feed_dict={X:x_test, W:weights[i].eval(), b:biases[i].eval()})
            print(yp)
            yp_tests.append(yp)
        
        # Get the predicted y array
        for i in range(len(x_test)):
            # Set up specific predicted class for each cell in the y_predicted_test
            predicted_class = 0
            max_score = yp_tests[0][i]

            # Get the highest score for each value in x test
            for index, test in enumerate(yp_tests):

                if test[i] > max_score:
                    # Change maximum score and set predicted class
                    max_score = test[i]
                    predicted_class = index

            # Append the predicted class to predicted test array
            y_predicted_test.append(predicted_class)

        return y_predicted_test


def hyperparameters_search():
    raise NotImplementedError


if __name__ == '__main__':
    hyperparameters_search()
