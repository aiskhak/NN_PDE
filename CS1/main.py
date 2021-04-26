# Third-party libraries
from time import perf_counter
import tensorflow as tf         # 1.15
import numpy as np
import os

# Modules
from figures import fields_fig_out, cost_fig_out, dev_fig_out, one_dim_fig_out, q_fig_out, q_dev_fig_out, sep_q_fig_out
from inout import set_paths, write_tec_var
from data_loader import load_data
from pde_solution import heat_model_fn
from ibcs import set_ibcs, grid_gen
from nn import init_w_b_fnn

# Launch on gpu '0' or cpu '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
if tf.test.gpu_device_name():
    print('GPU device is employed')
else:
    print('CPU device is employed')

print('####################################################', '\n',
      '# SETTING INPUT PARAMETERS AND READING DATA ########', '\n',
      '####################################################', '\n')

# GENERAL PARAMETERS
l = 1.0                     # Sizez of the domain [m]
n = 50                      # Number nodes in the whole field (including ghost ones for BCs)

# PARAMETERS FOR ITERATIONS AND OUTPUT CONTROL
max_it = 10000              # Maximum iterations for PDE solution
out_freq = 25               # Output frequency during PDE soution
eps = 1.e-6                 # Convergence criterion [K]

# NN HYPERPARAMETERS
lambd = 0.e-4               # Regularization parameter
learn_rate = 1.e-2          # Learning rate
lr_decay = 0.9999           # Learning rate decay parameter
drop_rate = 0.e0            # Dropout rate in fully connected layers
epochs = 100                # Number of training epochs
mod_sav_hold = 90           # Save the NN after this number of epochs
nn_out_freq = 100           # Frequency of NN information output

# FLAGS
is_data_ibc = True          # Use data for I&BCs; otherwise - linear initial distribution
is_solve_pde = True         # Solve PDEs using NN after training
is_plot_init = True         # Plot I&BCs in the beginning
is_plot_split = False       # Plot some of the split fields 3x3
is_train_nn = True          # Train a NN
is_read_nn_bef = False      # Read NN before training (if a pre-trained one exists)
is_read_nn_aft = True       # Read NN after training or use last step one
is_src_plot = True          # Plot source fields after training
is_tec_out = True           # Output tecplot files during PDE solution

# PATHS TO THE FOLDERS
work_dir, data_dir, fig_dir, csv_dir, tec_dir, nn_dir = set_paths()

# DATA
data, q_max, q_min, t_max, t_min = load_data(n, data_dir)

# GRID
h, x_split, y_split, x, y = grid_gen(n, l)

print('####################################################', '\n',
      '# SETTING AND OUTPUTTING I&BCs #####################', '\n',
      '####################################################', '\n')

# Set I&BCs
q_init, t_init, q_1, q_2, q_3 = set_ibcs(n, h, data.test.tar, data.test.inp, data.test.num_datasets, is_data_ibc, q_max, q_min)

# Tecplot files
res_plt_t_q = []
for k in range(data.test.num_datasets):
    
    # Open tecplot files
    res_plt_t_q.append(open(tec_dir + 'test/case_' + str(k+1) + '_t_q.plt', 'w'))
    
    # Tecplot files
    write_tec_var(n, t_init[k, :, :], q_init[k, :, :], y, x, 0, res_plt_t_q[k], 't', 'q')

    # Initial fields
    if is_plot_init:
        
        # Whole fields
        fields_fig_out(k, y, x, t_init[k, :, :], q_init[k, :, :], fig_dir + 'test/init/')

        # Separated sources
        sep_q_fig_out(k, y, x, q_1[k, :, :], q_2[k, :, :], q_3[k, :, :], fig_dir + 'test/init/sep/')
        
# Split fields 3x3 - for debugging (check correctness of splitting)
if is_plot_split:
    
    i = j = 0
    for k in range(0,(n-2)*(n-2)):
        
        if (k < (n-2)):             # first row 
        #if (k % (n-2) == 0):       # first column
        #if ((k+1) % (n-2) == 0):   # last column
        #if (k >= (n-2)*(n-3)):     # last row
            fields_fig_out(k, y_split[j,:], x_split[i,:], data.train.inp[k, :, :], data.train.tar[k, :, :], fig_dir + 'train/split/')
            #fields_fig_out(k, y_split[j,:], x_split[i,:], data.validation.inp[k, :, :], data.validation.tar[k, :, :], fig_dir + 'valid/split/')
        
        j += 1
        if ((k+1) % (n-2) == 0) and (k > 0):
            i += 1
            j = 0

print('####################################################', '\n',
      '# TENSORFLOW GRAPH #################################', '\n',
      '####################################################', '\n')

# Start the tensorflow session
sess = tf.compat.v1.Session()

# Weights and biases initialization for nn
weights, biases = init_w_b_fnn()

# Initialize constants
hh_tf = tf.constant(value=h*h, dtype=tf.float64, shape=())
c025_tf = tf.constant(value=0.25, dtype=tf.float64, shape=())
c05_tf = tf.constant(value=0.5, dtype=tf.float64, shape=())

# Placeholders
t_inp_ph = tf.compat.v1.placeholder(tf.float64, shape=[None, 3, 3], name='t_inp_ph')    # input temperature
t_tar_ph = tf.compat.v1.placeholder(tf.float64, shape=[None, 3, 3], name='t_tar_ph')    # target temperature
rate_ph = tf.compat.v1.placeholder(tf.float64, shape=(), name='rate_ph')                # dropout rate

# Update temperature
t_new, q_nn, q_ex, l2_q = heat_model_fn(t_inp_ph, rate_ph,
                                        c025_tf, hh_tf,
                                        weights, biases, 
                                        q_max, q_min, t_max, t_min)

# L2 regularization
reg = lambd*(tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + \
             tf.nn.l2_loss(weights['h3']) + tf.nn.l2_loss(weights['h4']) + \
             tf.nn.l2_loss(weights['h5']))

# Cost function
cost_train = tf.nn.l2_loss(t_new - tf.reshape(t_inp_ph[:, 1:-1, 1:-1], (-1, 1))) + reg
cost = tf.nn.l2_loss(t_new - tf.reshape(t_inp_ph[:, 1:-1, 1:-1], (-1, 1)))

# Adam optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learn_rate, 
                                             beta1=0.9, beta2=0.999, epsilon=1.e-7,
                                             use_locking=False, name='Adam').minimize(cost_train)

# Saver
saver = tf.compat.v1.train.Saver()

# Initialize state to ICs
sess.run(tf.compat.v1.global_variables_initializer())

# NN TRAINING
if is_train_nn:
    
    if is_read_nn_bef:

        print('####################################################', '\n',
              '# READING THE SAVED NN #############################', '\n',
              '####################################################', '\n')

        saver = tf.compat.v1.train.import_meta_graph(nn_dir + 'fnn.meta')
        saver.restore(sess, nn_dir + 'fnn')

    print('####################################################', '\n',
          '# NN TRAINING ######################################', '\n',
          '####################################################', '\n')

    best_epoch = 0
    best_cost = 1.e6
    best_l2_q = 1.e6
    cost_train = np.zeros(epochs, dtype=np.float64)
    l2_q_train = np.zeros(epochs, dtype=np.float64)
    cost_valid = np.zeros(epochs, dtype=np.float64)
    l2_q_valid = np.zeros(epochs, dtype=np.float64)
    batch_num = 10
    
    # Train NN 
    for i in range(epochs):

        for i_b in range(batch_num):
        
            # Train
            batch_train_x, batch_train_y = data.train.next_batch(int(data.train.num_datasets/batch_num))
            fd_train = {t_inp_ph: batch_train_x, t_tar_ph: batch_train_x, rate_ph: drop_rate}
            sess.run(optimizer, feed_dict=fd_train)
            
        # Evaluate training cost
        fd_evalu = {t_inp_ph: data.train.inp[:, :, :], t_tar_ph: data.train.inp[:, :, :], rate_ph: 0.0}
        cost_train[i] = sess.run(cost, feed_dict=fd_evalu)
        l2_q_train[i] = sess.run(l2_q, feed_dict=fd_evalu)

        # Evaluate validation cost
        fd_valid = {t_inp_ph: data.validation.inp[:, :, :], t_tar_ph: data.validation.inp[:, :, :], rate_ph: 0.0}
        cost_valid[i] = sess.run(cost, feed_dict=fd_valid)
        l2_q_valid[i] = sess.run(l2_q, feed_dict=fd_valid)

        if (cost_valid[i] < best_cost):
            
            # Print info
            best_cost = cost_valid[i]
            best_l2_q = l2_q_valid[i]
            best_epoch = i
            
            # Save nn
            if (i > mod_sav_hold):
                print('Saving the model. Epoch = ', i)
                saver.save(sess, nn_dir + 'fnn')

        # Screen output
        if i % nn_out_freq == 0 or i==(epochs-1):
            print ("Epoch:", '%04d' % (i+1), "Training cost =", "{:.12f}".format(cost_train[i]))
            print ("           ", "Validation cost =", "{:.12f}".format(cost_valid[i]))
            print ("           ", "Training L2 norm =", "{:.12f}".format(l2_q_train[i]))
            print ("           ", "Validation L2 norm =", "{:.12f}".format(l2_q_valid[i]))
        
        # Learn rat decay
        learn_rate *= lr_decay

    # Information on the best epoch
    print ("Best epoch = ", best_epoch)
    print ("Best validation cost = ", best_cost)
    print ("Validation L2 norm at that epoch = ", best_l2_q)
    print ("Training cost at that epoch = ", cost_train[best_epoch])
    print ("Training L2 norm at that epoch = ", l2_q_train[best_epoch])

    # Plot cost and RMSE
    cost_fig_out(epochs, cost_train, cost_valid, l2_q_train, l2_q_valid, fig_dir + 'epochs/')
    
if is_read_nn_aft:

    print('####################################################', '\n',
          '# READING THE SAVED NN #############################', '\n',
          '####################################################', '\n')

    saver = tf.compat.v1.train.import_meta_graph(nn_dir + 'fnn.meta')
    saver.restore(sess, nn_dir + 'fnn')

if is_src_plot:

    print('####################################################', '\n',
          '# PLOTTING SOURCES FIELDS ##########################', '\n',
          '####################################################', '\n')
    
    fd_evalu = {t_inp_ph: data.train.inp[:, :, :], rate_ph: 0.0}
    
    q_nn_plot = q_nn.eval(session=sess, feed_dict=fd_evalu)
    q_nn_plot = np.reshape(q_nn_plot, (data.train.num_datasets,-1))

    q_ex_plot = sess.run(q_ex, feed_dict=fd_evalu)
    q_ex_plot = np.reshape(q_ex_plot, (data.train.num_datasets,-1))

    q_split_plot = np.reshape(data.train.tar[:, 1:-1, 1:-1], (data.train.num_datasets,-1))
    q_split_plot = q_split_plot*(q_max - q_min) + q_min

    t_new_plot = sess.run(t_new, feed_dict=fd_evalu)
    t_new_plot = np.reshape(t_new_plot, (data.train.num_datasets,-1))

    t_split_plot = np.reshape(data.train.inp[:, 1:-1, 1:-1], (data.train.num_datasets,-1))

    q_dev_fig_out(1, q_nn_plot, q_split_plot, fig_dir  + 'epochs/q/dev/')
    q_dev_fig_out(2, q_ex_plot, q_split_plot, fig_dir  + 'epochs/q/dev/')
    q_dev_fig_out(3, t_new_plot, t_split_plot, fig_dir  + 'epochs/t/dev/')

if is_solve_pde:

    print('####################################################', '\n',
          '# SOLUTION OF PDE ##################################', '\n',
          '####################################################', '\n')

    # Initialize variables
    t_pde = tf.Variable(initial_value=t_init, dtype=tf.float64)

    # Update temperature
    t_new_pde_, q_nn_pde_, q_ex_pde_, l2_q_pde = heat_model_fn(t_pde, 0.0,
                                                               c025_tf, hh_tf,
                                                               weights, biases,
                                                               q_max, q_min, t_max, t_min)
    t_new_pde = tf.reshape(t_new_pde_, (-1, (n-2), (n-2)))
    q_nn_pde = tf.reshape(q_nn_pde_, (-1, (n-2), (n-2)))
    q_ex_pde = tf.reshape(q_ex_pde_, (-1, (n-2), (n-2)))

    # Update step
    step_temp = tf.group(t_pde[:, 1:-1, 1:-1].assign(t_new_pde))

    # Convergence
    abs_er = tf.abs(t_new_pde - t_pde[:, 1:-1, 1:-1])

    # Initialize state to initial conditions
    sess.run(tf.initialize_variables([t_pde]))

    # Arrays for errors and cost
    abs_er_max = np.zeros(shape=(max_it, data.test.num_datasets), dtype=np.float64)

    # Time iterations
    start_time_it = perf_counter()
    it = 0
    max_er = 1.e6
    while (it < max_it) and (max_er > eps):

        # PDE solution
        step_temp.run(session=sess)

        # Error
        for k in range(data.test.num_datasets):
            abs_er_max[it, k] = np.max(sess.run(abs_er)[k, :, :])
        max_er = np.max(abs_er_max[it, :])
        
        # Next iteration
        it += 1

        # Information
        if (it % out_freq == 0):
            print('--------------------------------------------', '\n',
                  'Iteration =', it, '\n',
                  'Max error = ', max_er, '\n',
                  'L2(q) = ', sess.run(l2_q_pde))
            
            # Tecplot files
            if is_tec_out:
                for k in range(data.test.num_datasets):
                    write_tec_var(n-2, sess.run(t_pde[k, 1:-1, 1:-1]), sess.run(q_nn_pde[k, :, :]), x[1:-1], y[1:-1], it, res_plt_t_q[k], 'p', 'q')
    
    # Timer
    end_time_it = perf_counter()
    print('--------------------------------------------', '\n',
          'SOLUTION OF PDE IS FINISHED. TIME CONSUMED (S) =', end_time_it-start_time_it, '\n',
          'NUMBER OF ITERATIONS =', it)

    print('####################################################', '\n',
          '# OUTPUT RESULTS ###################################', '\n',
          '####################################################', '\n')

    # Last step values
    t_fin = sess.run(t_pde)
    q_fin = sess.run(q_nn_pde)

    # Deviations for all datasets
    dev_fig_out(np.reshape(t_fin, (n*n*data.test.num_datasets)), np.reshape(data.test.inp, (n*n*data.test.num_datasets)), fig_dir  + 'test/t_dev/')
    q_dev_fig_out(1000, np.reshape(q_fin, ((n-2)*(n-2)*data.test.num_datasets)), np.reshape(data.test.tar[:,1:-1,1:-1]*(q_max-q_min) + q_min, ((n-2)*(n-2)*data.test.num_datasets)), fig_dir  + 'test/q_dev/')

    for k in range(data.test.num_datasets):
    
        # Tecplot files
        write_tec_var(n-2, t_fin[k, 1:-1, 1:-1], q_fin[k, :, :], x[1:-1], y[1:-1], it, res_plt_t_q[k], 'p', 'q')
   
        # Close tecplot files
        res_plt_t_q[k].close()
    
        # Norms
        t_norm = np.amax(np.abs(t_fin[k, :, :] - data.test.inp[k, :, :]))
        q_norm = np.amax(np.abs(q_fin[k, :, :] - data.test.tar[k, 1:-1, 1:-1]*(q_max-q_min) + q_min))
        max = np.amax(q_fin)
        min = np.amin(q_fin)
        norms = np.array([t_norm, q_norm, max, min])
        np.savetxt(csv_dir + 'case_' + str(k+1) + '_norms.dat', norms, delimiter = " ")

        # Fields
        #fields_fig_out(k, x[1:-1], y[1:-1], t_fin[k, 1:-1, 1:-1], q_fin[k, :, :], fig_dir + 'test/res/fields/')
        q_fig_out(k, x[1:-1], y[1:-1], q_fin[k, :, :], data.test.tar[k, 1:-1, 1:-1]*(q_max-q_min) + q_min, fig_dir + 'test/res/q/')
        q_fig_out(k, x[1:-1], y[1:-1], t_fin[k, 1:-1, 1:-1], data.test.inp[k, 1:-1, 1:-1], fig_dir + 'test/res/t/')

        # 1D results
        one_dim_fig_out(k, it, abs_er_max, eps, fig_dir + 'test/res/1D/')

# Close Tensorflow session
sess.close()