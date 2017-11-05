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

  # Weights
  W_conv1 = tf.Variable(tf.random_normal([5, 5, 3, 64]), name='W_conv1')
  W_conv2 = tf.Variable(tf.random_normal([5, 5, 64, 64]), name='W_conv2')
  W_conv3 = tf.Variable(tf.random_normal([5, 5, 64, 128]), name='W_conv3')
  W_conv4 = tf.Variable(tf.random_normal([5, 5, 128, 128]), name='W_conv4')
  W_fc = tf.Variable(tf.random_normal([4*4*128, 4*4*256]), name='W_fc')
  W_fc1 = tf.Variable(tf.random_normal([4*4*256, 1024]), name='W_fc1')
  w_output = tf.Variable(tf.random_normal([1024, NUM_CLASSES]), name='w_output')
  
  # Biases
  b_conv1 = tf.Variable(tf.random_normal([64]), name='b_conv1')
  b_conv2 = tf.Variable(tf.random_normal([64]), name='b_conv2')
  b_conv3 = tf.Variable(tf.random_normal([128]), name='b_conv3')
  b_conv4 = tf.Variable(tf.random_normal([128]), name='b_conv4')
  b_fc = tf.Variable(tf.random_normal([4*4*256]), name='b_fc')
  b_fc1 = tf.Variable(tf.random_normal([1024]), name='b_fc1')
  b_output = tf.Variable(tf.random_normal([NUM_CLASSES]), name='b_output')
  '''
          for 4096
Step: 0 Loss: 1.25985e+09 Batch accuracy: 0.11
Step: 50 Loss: 1.40388e+08 Batch accuracy: 0.23
Step: 100 Loss: 1.29185e+08 Batch accuracy: 0.26
Step: 150 Loss: 9.15848e+07 Batch accuracy: 0.31
Step: 200 Loss: 7.04282e+07 Batch accuracy: 0.39
Step: 250 Loss: 8.63989e+07 Batch accuracy: 0.22
Step: 300 Loss: 7.16035e+07 Batch accuracy: 0.33
Step: 350 Loss: 7.16511e+07 Batch accuracy: 0.34
Step: 400 Loss: 5.93483e+07 Batch accuracy: 0.37
Step: 450 Loss: 4.84606e+07 Batch accuracy: 0.41
Epoch: 0 Average cost: 109582205.4
Step: 0 Loss: 4.62275e+07 Batch accuracy: 0.33

        '''
  '''
          for 2048
    Step: 0 Loss: 8.05779e+08 Batch accuracy: 0.16
    Step: 50 Loss: 1.16221e+08 Batch accuracy: 0.26
    Step: 100 Loss: 7.90658e+07 Batch accuracy: 0.34
    Step: 150 Loss: 8.58765e+07 Batch accuracy: 0.3
    Step: 200 Loss: 8.9483e+07 Batch accuracy: 0.22
    Step: 250 Loss: 5.55266e+07 Batch accuracy: 0.3
    Step: 300 Loss: 6.65387e+07 Batch accuracy: 0.3
    Step: 350 Loss: 5.89892e+07 Batch accuracy: 0.32
    Step: 400 Loss: 4.82905e+07 Batch accuracy: 0.34
    Step: 450 Loss: 5.80454e+07 Batch accuracy: 0.28
    Epoch: 0 Average cost: 91305863.728
    Step: 0 Loss: 4.46995e+07 Batch accuracy: 0.41
    Step: 50 Loss: 4.95835e+07 Batch accuracy: 0.29
    Step: 100 Loss: 4.57668e+07 Batch accuracy: 0.31
    Step: 150 Loss: 4.38167e+07 Batch accuracy: 0.35

        '''
  '''
        for 1024
    Step: 0 Loss: 5.34815e+08 Batch accuracy: 0.1
    Step: 50 Loss: 9.02175e+07 Batch accuracy: 0.19
    Step: 100 Loss: 5.78763e+07 Batch accuracy: 0.26
    Step: 150 Loss: 4.48077e+07 Batch accuracy: 0.28
    Step: 200 Loss: 4.7721e+07 Batch accuracy: 0.33
    Step: 250 Loss: 4.21907e+07 Batch accuracy: 0.27
    Step: 300 Loss: 3.17977e+07 Batch accuracy: 0.38
    Step: 350 Loss: 3.94594e+07 Batch accuracy: 0.21
    Step: 400 Loss: 3.31846e+07 Batch accuracy: 0.32
    Step: 450 Loss: 3.49477e+07 Batch accuracy: 0.34
    Epoch: 0 Average cost: 56856133.112
    Step: 0 Loss: 2.99255e+07 Batch accuracy: 0.36
    Step: 50 Loss: 3.16833e+07 Batch accuracy: 0.33
    Step: 100 Loss: 2.87861e+07 Batch accuracy: 0.28
      '''
  '''
        for 512
    Step: 0 Loss: 2.62955e+08 Batch accuracy: 0.16
    Step: 50 Loss: 5.49145e+07 Batch accuracy: 0.22
    Step: 100 Loss: 4.74106e+07 Batch accuracy: 0.2
    Step: 150 Loss: 3.35041e+07 Batch accuracy: 0.24
    Step: 200 Loss: 2.91453e+07 Batch accuracy: 0.34
    Step: 250 Loss: 3.11051e+07 Batch accuracy: 0.26
    Step: 300 Loss: 2.07875e+07 Batch accuracy: 0.27
    Step: 350 Loss: 2.20899e+07 Batch accuracy: 0.33
    Step: 400 Loss: 2.0257e+07 Batch accuracy: 0.32
    Step: 450 Loss: 1.83876e+07 Batch accuracy: 0.27
    Epoch: 0 Average cost: 36127248.866
    Step: 0 Loss: 1.68179e+07 Batch accuracy: 0.3
    Step: 50 Loss: 1.71867e+07 Batch accuracy: 0.29
    Step: 100 Loss: 1.56044e+07 Batch accuracy: 0.36
      '''
  '''
      for 256
     Step: 0 Loss: 4.13503e+08 Batch accuracy: 0.05
    Step: 50 Loss: 4.13816e+07 Batch accuracy: 0.15
    Step: 100 Loss: 3.11436e+07 Batch accuracy: 0.21
    Step: 150 Loss: 2.53437e+07 Batch accuracy: 0.26
    Step: 200 Loss: 2.21766e+07 Batch accuracy: 0.18
    Step: 250 Loss: 1.98106e+07 Batch accuracy: 0.16
    Step: 300 Loss: 1.81124e+07 Batch accuracy: 0.28
    Step: 350 Loss: 1.83526e+07 Batch accuracy: 0.3
    Step: 400 Loss: 1.30438e+07 Batch accuracy: 0.35
    Step: 450 Loss: 1.54771e+07 Batch accuracy: 0.25
    Epoch: 0 Average cost: 31298666.026
    Step: 0 Loss: 1.22778e+07 Batch accuracy: 0.3
    Step: 50 Loss: 1.49523e+07 Batch accuracy: 0.29
    Step: 100 Loss: 1.3439e+07 Batch accuracy: 0.35
    Step: 150 Loss: 1.10937e+07 Batch accuracy: 0.33
    '''
  '''
      for 128
    Step: 0 Loss: 2.26603e+08 Batch accuracy: 0.1
    Step: 50 Loss: 2.70491e+07 Batch accuracy: 0.18
    Step: 100 Loss: 2.10527e+07 Batch accuracy: 0.16
    Step: 150 Loss: 1.59699e+07 Batch accuracy: 0.17
    Step: 200 Loss: 1.37864e+07 Batch accuracy: 0.26
    Step: 250 Loss: 1.20644e+07 Batch accuracy: 0.21
    Step: 300 Loss: 1.10787e+07 Batch accuracy: 0.29
    Step: 350 Loss: 9.83537e+06 Batch accuracy: 0.23
    Step: 400 Loss: 9.15449e+06 Batch accuracy: 0.17
    Step: 450 Loss: 9.48555e+06 Batch accuracy: 0.27
    Epoch: 0 Average cost: 17969936.57
    Step: 0 Loss: 7.72597e+06 Batch accuracy: 0.35
    Step: 50 Loss: 6.19504e+06 Batch accuracy: 0.37
    Step: 100 Loss: 6.55516e+06 Batch accuracy: 0.29
    Step: 150 Loss: 8.90248e+06 Batch accuracy: 0.26
    Step: 250 Loss: 6.30182e+06 Batch accuracy: 0.28
    Step: 200 Loss: 6.88812e+06 Batch accuracy: 0.32
    Step: 300 Loss: 5.28157e+06 Batch accuracy: 0.38
    Step: 350 Loss: 5.81831e+06 Batch accuracy: 0.32
    Step: 400 Loss: 6.0897e+06 Batch accuracy: 0.3
    Step: 450 Loss: 5.12045e+06 Batch accuracy: 0.36
    Epoch: 1 Average cost: 6419287.9615
    Step: 0 Loss: 5.1866e+06 Batch accuracy: 0.34
    '''
  '''
    for 64
    Step: 0 Loss: 1.68247e+08 Batch accuracy: 0.1
    Step: 50 Loss: 1.90294e+07 Batch accuracy: 0.16
    Step: 100 Loss: 1.40491e+07 Batch accuracy: 0.14
    Step: 150 Loss: 1.15751e+07 Batch accuracy: 0.19
    Step: 200 Loss: 9.83542e+06 Batch accuracy: 0.25
    Step: 250 Loss: 8.17271e+06 Batch accuracy: 0.24
    Step: 300 Loss: 8.09242e+06 Batch accuracy: 0.26
    Step: 350 Loss: 6.76577e+06 Batch accuracy: 0.28
    Step: 400 Loss: 6.52733e+06 Batch accuracy: 0.25
    Step: 450 Loss: 6.66111e+06 Batch accuracy: 0.3
    Epoch: 0 Average cost: 13015029.433
    Step: 0 Loss: 3.69151e+06 Batch accuracy: 0.34
    Step: 50 Loss: 4.67989e+06 Batch accuracy: 0.33
    Step: 100 Loss: 6.78497e+06 Batch accuracy: 0.27
    Step: 150 Loss: 8.90248e+06 Batch accuracy: 0.26
  '''
  '''
    for 32
    Step: 0 Loss: 1.64833e+08 Batch accuracy: 0.07
    Step: 50 Loss: 2.00059e+07 Batch accuracy: 0.12
    Step: 100 Loss: 1.12882e+07 Batch accuracy: 0.19
    Step: 150 Loss: 8.60072e+06 Batch accuracy: 0.17
    Step: 200 Loss: 5.87024e+06 Batch accuracy: 0.16
    Step: 250 Loss: 86749.4 Batch accuracy: 0.06
    Step: 300 Loss: 159200.0 Batch accuracy: 0.08
    Step: 350 Loss: 101202.0 Batch accuracy: 0.07
    Step: 400 Loss: 53752.2 Batch accuracy: 0.08
    Step: 450 Loss: 2.5388 Batch accuracy: 0.1
    Epoch: 0 Average cost: 8394796.92504
    Step: 0 Loss: 11287.2 Batch accuracy: 0.06
  '''





  # Layer 1
  xdrop = tf.nn.dropout(x, dropout_kept_prob)
  conv1 = conv_2d(xdrop, W_conv1) + b_conv1
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
  conv3 = tf.nn.dropout(conv3, 0.5)
#  # Layer 4
  conv4 = conv_2d(conv3, W_conv4) + b_conv4
  conv4 = tf.nn.relu(conv4)  # Skip connection from Layer 3
  conv4 = maxpool_2d(conv4)

  # Fully Connected Layer 4

  fc = tf.reshape(conv4, [-1, 4*4*128])
  fc = tf.nn.relu(tf.matmul(fc, W_fc) + b_fc)

  # Fully Connected Layer 5
  fc1 = tf.reshape(fc, [-1, 4*4*256])
  fc1 = tf.nn.relu(tf.matmul(fc1, W_fc1) + b_fc1)

  # Apply dropout
  #fc1 = tf.nn.dropout(fc1, dropout_kept_prob)

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

    keep_prob = 0.8
    lr = 0.001

    output = net(x_input, True, keep_prob)
    print(output.shape)

    # Get loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    epochs = 30

    with tf.Session() as sess:
      sess.run(init)
      
      correct = tf.equal(tf.argmax(output, 1), tf.argmax(y,1))
      accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

      for epoch in range(epochs):
        avg_cost = 0
        num_batches = num_samples // BATCH_SIZE    

        for iteration in range(num_batches):
          x_batch, y_batch = cifar10_train.get_next_batch()
          _, cost = sess.run([optimizer, loss], feed_dict={x: x_batch, y: y_batch})
          avg_cost += cost / num_batches

          if iteration % 50 == 0:
          	_loss, batch_acc = sess.run([loss, accuracy], feed_dict={x:x_batch, y:y_batch})
          	print("Step:", iteration, "Loss:", _loss, "Batch accuracy:", batch_acc)

            

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

    keep_prob = 1 # no dropout for test

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
        'W_conv4:0': sess.run('W_conv4:0'),
        'b_conv4:0': sess.run('b_conv4:0'),
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
