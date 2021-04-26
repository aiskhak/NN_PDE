####################################################
# SOLUTION OF THE NS EQS USING TENSORFLOW ##########
####################################################

# Third-party libraries
import tensorflow as tf

# Libraries
from comp_conv import kernel_1
from nn import fnn

# Heat model
def heat_model_fn(t, rate,
                  c025_tf, hh_tf,
                  weights, biases,
                  q_max, q_min, t_max, t_min):

    # sources predicted by nn
    t_inp = tf.reshape(t[:, 1:-1, 1:-1], (-1, 1))
    t_inp_norm = (t_inp - t_min) / (t_max - t_min)
    q_nn = fnn(t_inp_norm, rate, weights, biases)*(q_max - q_min) + q_min

    # Exact sources
    q_ex = -4.e2*tf.math.sqrt(t_inp) + 1.e1*(t_inp - 1.e2) + 1.e-8*tf.pow(t_inp, 4)

    # RMSE
    l2_q = tf.nn.l2_loss(q_ex - q_nn)

    # new temperature
    t_ker = kernel_1(t)
    t_conv = tf.reshape(t_ker, (-1, 1))
    t_new = c025_tf*tf.math.add(t_conv, -hh_tf*q_nn)

    return t_new, q_nn, q_ex, l2_q


    