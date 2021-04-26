####################################################
# NEURAL NETWORKS ##################################
####################################################

# Third-party libraries
import tensorflow as tf

# FNN
def fnn(x, drop_rate, weights, biases):

    # Fully-connected layer + Activation + Dropout 1
    layer_fc1 = tf.matmul(x, weights['h1']) + biases['h1']
    layer_fc1 = tf.nn.sigmoid(layer_fc1)
    layer_fc1 = tf.nn.dropout(layer_fc1, rate=drop_rate)
    
    # Fully-connected layer + Activation + Dropout 2
    layer_fc2 = tf.matmul(layer_fc1, weights['h2']) + biases['h2']
    layer_fc2 = tf.nn.sigmoid(layer_fc2)
    layer_fc2 = tf.nn.dropout(layer_fc2, rate=drop_rate)
    
    # Fully-connected layer + Activation + Dropout 3
    layer_fc3 = tf.matmul(layer_fc2, weights['h3']) + biases['h3']
    layer_fc3 = tf.nn.sigmoid(layer_fc3)
    layer_fc3 = tf.nn.dropout(layer_fc3, rate=drop_rate)
    
    # Fully-connected layer + Activation + Dropout 4
    layer_fc4 = tf.matmul(layer_fc3, weights['h4']) + biases['h4']
    layer_fc4 = tf.nn.sigmoid(layer_fc4)
    layer_fc4 = tf.nn.dropout(layer_fc4, rate=drop_rate)

    # Fully-connected layer
    layer_fc5 = tf.matmul(layer_fc4, weights['h5']) + biases['h5']
    layer_fc5 = tf.nn.sigmoid(layer_fc5)

    return layer_fc5

# Weights and biases initialization for fnn
def init_w_b_fnn():

    n_1 = 8
    n_2 = 16
    n_3 = 32
    n_4 = 16
    w1_in = tf.contrib.layers.xavier_initializer()
    w2_in = tf.contrib.layers.xavier_initializer() 
    w3_in = tf.contrib.layers.xavier_initializer() 
    w4_in = tf.contrib.layers.xavier_initializer() 
    w5_in = tf.contrib.layers.xavier_initializer() 
    weights = {'h1': tf.compat.v1.get_variable("w1", shape=[1, n_1], initializer=w1_in, dtype=tf.float64),
               'h2': tf.compat.v1.get_variable("w2", shape=[n_1, n_2], initializer=w2_in, dtype=tf.float64),
               'h3': tf.compat.v1.get_variable("w3", shape=[n_2, n_3], initializer=w3_in, dtype=tf.float64),
               'h4': tf.compat.v1.get_variable("w4", shape=[n_3, n_4], initializer=w4_in, dtype=tf.float64),
               'h5': tf.compat.v1.get_variable("w5", shape=[n_4, 1], initializer=w5_in, dtype=tf.float64)} 
    biases = {'h1': tf.Variable(tf.random.truncated_normal(shape=[n_1], mean=0.0, stddev=1.0, dtype=tf.float64)),
              'h2': tf.Variable(tf.random.truncated_normal(shape=[n_2], mean=0.0, stddev=1.0, dtype=tf.float64)),
              'h3': tf.Variable(tf.random.truncated_normal(shape=[n_3], mean=0.0, stddev=1.0, dtype=tf.float64)),
              'h4': tf.Variable(tf.random.truncated_normal(shape=[n_4], mean=0.0, stddev=1.0, dtype=tf.float64)),
              'h5': tf.Variable(tf.random.truncated_normal(shape=[1], mean=0.0, stddev=1.0, dtype=tf.float64))}

    return weights, biases