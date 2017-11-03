from ops import *
import timeit
from cifar10 import Cifar10

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
BATCH_SIZE = 10

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

  # Weights
  W_conv1 = tf.Variable(tf.random_normal([5, 5, 3, 64]), name='W_conv1')
  W_conv2 = tf.Variable(tf.random_normal([5, 5, 64, 64]), name='W_conv2')
  W_conv3 = tf.Variable(tf.random_normal([5, 5, 64, 128]), name='W_conv3')
  W_fc = tf.Variable(tf.random_normal([4*4*128, 512]), name='W_fc')
  W_fc1 = tf.Variable(tf.random_normal([4*4*128, 256]), name='W_fc1')
  w_output = tf.Variable(tf.random_normal([256, NUM_CLASSES]), name='w_output')
  
  # Biases
  b_conv1 = tf.Variable(tf.random_normal([64]), name='b_conv1')
  b_conv2 = tf.Variable(tf.random_normal([64]), name='b_conv2')
  b_conv3 = tf.Variable(tf.random_normal([128]), name='b_conv3')
  b_fc = tf.Variable(tf.random_normal([512]), name='b_fc')
  b_fc1 = tf.Variable(tf.random_normal([256]), name='b_fc1')
  b_output = tf.Variable(tf.random_normal([NUM_CLASSES]), name='b_output')
  







  # Layer 1
  conv1 = conv_2d(x, W_conv1) + b_conv1
  conv1 = tf.nn.relu(conv1)
  conv1 = maxpool_2d(conv1)

  # Layer 2
  conv2 = conv_2d(conv1, W_conv2) + b_conv2
  conv2 = batch_norm(conv2, is_training) + conv1
  conv2 = tf.nn.relu(conv2)
  conv2 = maxpool_2d(conv2)
  
  # Layer 3
  conv3 = conv_2d(conv2, W_conv3) + b_conv3
  conv3 = tf.nn.relu(conv3)

#  # Layer 4
#  conv4 = conv_2d(conv3, weights['W_conv4']) + biases['b_conv4']
#  conv4 = tf.nn.relu(conv4)  # Skip connection from Layer 3
#  conv4 = maxpool_2d(conv4)

  # Fully Connected Layer 4
  fc = tf.reshape(conv3, [-1, 4*4*128])
  fc = tf.nn.relu(tf.matmul(fc, W_fc) + b_fc)

  # Fully Connected Layer 5
  fc1 = tf.reshape(fc, [-1, 4*4*128])
  fc1 = tf.nn.relu(tf.matmul(fc1, W_fc1) + b_fc1)

  # Apply dropout
  fc1 = tf.nn.dropout(fc1, dropout_kept_prob)

  # Output
  output = tf.matmul(fc1, w_output) + b_output

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

  with tf.Graph().as_default() as g:

    # Load CIFAR-10 training data
    cifar10_train = Cifar10(batch_size=BATCH_SIZE, one_hot=True, test=False, shuffle=True)
    cifar10_train_images = cifar10_train._images # 50000
    cifar10_train_labels = cifar10_train._labels # 50000

    num_samples = cifar10_train.num_samples

    # Define placeholder variable for input images
    x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], name="x")
    
    # Define placeholder variable for true labels
    y = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES], name="y")

    x_input = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], name="input")

    keep_prob = 0.5
    lr = 0.0001

    output = net(x_input, True, keep_prob)
    print(output.shape)

    # Get loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    epochs = 1

    with tf.Session() as sess:
      sess.run(init)

      for epoch in range(epochs):
        avg_cost = 0
        num_batches = num_samples // BATCH_SIZE    

        for iteration in range(num_batches):
          x_batch, y_batch = cifar10_train.get_next_batch()
          _, cost = sess.run([optimizer, loss], feed_dict={x: x_batch, y: y_batch})
          avg_cost += cost / num_batches

          if iteration % 50 == 0:
            print('Step:', iteration, 'Cost:', cost)

        print('Epoch:', epoch, 'Average cost:', avg_cost)

      saver.save(sess, './my_test_model')


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

  with tf.Graph().as_default() as g:

    # Define placeholder variable for input images
    x = tf.placeholder(tf.float32, shape=[None, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], name="x")

    x_input = tf.reshape(x, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS], name="input")

    keep_prob = 0.5

    output = net(x_input, False, keep_prob)

    # Load the saved model
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

      saver = tf.train.Saver()
      saver.restore(sess, tf.train.latest_checkpoint('.'))

      feed_dict = {
        x: cifar10_test_images,
        'W_conv1:0': sess.run('W_conv1:0'),
        'b_conv1:0': sess.run('b_conv1:0'),
        'W_conv2:0': sess.run('W_conv2:0'),
        'b_conv2:0': sess.run('b_conv2:0'),
        'W_conv3:0': sess.run('W_conv3:0'),
        'b_conv3:0': sess.run('b_conv3:0'),
        'W_fc:0': sess.run('W_fc:0'),
        'b_fc:0': sess.run('b_fc:0'),
        'W_fc1:0': sess.run('W_fc1:0'),
        'b_fc1:0': sess.run('b_fc1:0')
      }

      labels = []

      pred = sess.run(output, feed_dict=feed_dict)

      for array in pred:
        pred_class = 0
        max_score = array[0]

        for i, num in enumerate(array):
          if num > max_score:
            max_score = num
            pred_class = i

        labels.append([pred_class])

      print(labels)
      return np.array(labels)
