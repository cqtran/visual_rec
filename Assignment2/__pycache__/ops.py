# Reference: https://github.com/openai/improved-gan/blob/master/imagenet/ops.py
import tensorflow as tf
import numpy as np


# TODO: Optionally create utility functions for convolution, fully connected layers here

def batch_norm(input, is_training, momentum=0.9, epsilon=1e-5, in_place_update=True, name="batch_norm"):
  # Example batch_norm usage:
  # bn0 = batch_norm(conv1, is_training=is_training, name='bn0')
  if in_place_update:
    return tf.contrib.layers.batch_norm(input,
                                        decay=momentum,
                                        center=True,
                                        scale=True,
                                        epsilon=epsilon,
                                        updates_collections=None,
                                        is_training=is_training,
                                        scope=name)
  else:
    return tf.contrib.layers.batch_norm(input,
                                        decay=momentum,
                                        center=True,
                                        scale=True,
                                        epsilon=epsilon,
                                        is_training=is_training,
                                        scope=name)

def conv_2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool_2d(x):
  return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
