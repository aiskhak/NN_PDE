####################################################
# COMPUTATIONAL CONVENIENCE FUNCTIONS ##############
####################################################

# Third-party libraries
import tensorflow as tf
import numpy as np

# Transform a 2D array into a convolution kernel
def make_kernel(a):
    a = tf.expand_dims(a, axis=2)
    a = tf.expand_dims(a, axis=3)
    return a

# Convolution operation
def conv_valid(x, kernel):
    x = tf.expand_dims(x, 3)
    y = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding='VALID')
    return y[:, :, :, 0]

# Kernel #1
def kernel_1(x):
    filter = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]], dtype=np.float64)
    return conv_valid(x, make_kernel(filter))