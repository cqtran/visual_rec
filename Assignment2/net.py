from ops import *
import timeit
from cifar10 import Cifar10

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
BATCH_SIZE = 100


def net(x, is_training, dropout_kept_prob):
  # TODO: Write your network architecture here
  # Or you can write it inside train() function
  # Requirements:
  # - At least 5 layers in total
  # - At least 1 fully connected and 1 convolutional layer
  # - At least one maxpool layers
  # - At least one batch norm
  # - At least one skip connection
  # - Use dropout

  # Initialize weights dictionary
  weights = {'W_conv1':tf.Variable(tf.random_normal([5, 5, 3, 64])),
             'W_conv2':tf.Variable(tf.random_normal([5, 5, 64, 64])),
             'W_conv3':tf.Variable(tf.random_normal([5, 5, 64, 128])),
             'W_conv4':tf.Variable(tf.random_normal([3, 3, 128, 128])),
             'W_fc':tf.Variable(tf.random_normal([4*4*128, 384])),
             'output':tf.Variable(tf.random_normal([384, NUM_CLASSES]))}

  # Initialize biases dictionary
  biases = {'b_conv1':tf.Variable(tf.random_normal([64])),
            'b_conv2':tf.Variable(tf.random_normal([64])),
            'b_conv3':tf.Variable(tf.random_normal([128])),
            'b_conv4':tf.Variable(tf.random_normal([128])),
            'b_fc':tf.Variable(tf.random_normal([384])),
            'output':tf.Variable(tf.random_normal([NUM_CLASSES]))}

  # Layer 1
  conv1 = conv_2d(x, weights['W_conv1']) + biases['b_conv1'] 
  conv1 = tf.nn.relu(conv1)
  conv1 = maxpool_2d(conv1)

  # Layer 2
  conv2 = conv_2d(conv1, weights['W_conv2']) + biases['b_conv2']
  conv2 = batch_norm(conv2, is_training) + conv1
  conv2 = tf.nn.relu(conv2)
  conv2 = maxpool_2d(conv2)
  
  # Layer 3
  conv3 = conv_2d(conv2, weights['W_conv3']) + biases['b_conv3']
  conv3 = tf.nn.relu(conv3)

  # Layer 4
  conv4 = conv_2d(conv3, weights['W_conv4']) + biases['b_conv4']
  conv4 = tf.nn.relu(conv4)  # Skip connection from Layer 3
  conv4 = maxpool_2d(conv4)

  # Fully Connected Layer 5
  fc = tf.reshape(conv4, [-1, 4*4*128])
  fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

  # Apply dropout
  fc = tf.nn.dropout(fc, dropout_kept_prob)

  # Output
  output = tf.matmul(fc, weights['output']) + biases['output']
  print(output.shape)
  return output

def train():
  # Always use tf.reset_default_graph() to avoid error
  tf.reset_default_graph()
  # TODO: Write your training code here
  # - Create placeholder for inputs, training boolean, dropout keep probablity
  # - Construct your model
  # - Create loss and training op
  # - Run training
  # AS IT WILL TAKE VERY LONG ON CIFAR10 DATASET TO TRAIN
  # YOU SHOULD USE tf.train.Saver() TO SAVE YOUR MODEL AFTER TRAINING
  # AT TEST TIME, LOAD THE MODEL AND RUN TEST ON THE TEST SET

  # Load CIFAR-10 training data
  cifar10_train = Cifar10(batch_size=BATCH_SIZE, one_hot=True, test=False, shuffle=True)
  cifar10_train_images = cifar10_train._images # 50000
  cifar10_train_labels = cifar10_train._labels # 50000

  num_samples = cifar10_train.num_samples

  # Define placeholder variable for input images
  x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], name="x")
  
  # Define placeholder variable for true labels
  y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES], name="y")

  x_input = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], name="input")

  keep_prob = 0.5
  lr = 0.0001

  output = net(x_input, True, keep_prob)

  # Get loss
  '''
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
  optimizer = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss)

  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  epochs = 10  

  with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
      num_batches = num_samples // BATCH_SIZE
      epoch_loss = 0
      for iteration in range(num_batches):
        x_batch, y_batch = cifar10_train.get_next_batch()
        _loss, batch_acc = sess.run([optimizer, loss], feed_dict={x: x_batch, y: y_batch})
        saver.save(sess, save_path='/Users/sharidanbarboza/328_notebooks/Assignment2/')
        epoch_loss += _loss

        print('Epoch', epoch, 'Loss:', epoch_loss)

    correct = tf.equal(tf.argmax(output, 1), tf.argmax(y,1))
    acc = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy:', acc.eval({x:cifar10_test_images, y:cifar10_train_labels}))

  raise NotImplementedError
  '''

def test(cifar10_test_images):
  # Always use tf.reset_default_graph() to avoid error
  tf.reset_default_graph()
  # TODO: Write your testing code here
  # - Create placeholder for inputs, training boolean, dropout keep probablity
  # - Construct your model
  # (Above 2 steps should be the same as in train function)
  # - Create label prediction tensor
  # - Run testing
  # DO NOT RUN TRAINING HERE!
  # LOAD THE MODEL AND RUN TEST ON THE TEST SET
  raise NotImplementedError
