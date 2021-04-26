# Third-party libraries
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import random_seed

# DataSet class
class DataSet(object):
  
    def __init__(self, inp, tar, seed=None):
        seed1, seed2 = random_seed.get_seed(seed)
        np.random.seed(seed1 if seed is None else seed2)
        assert inp.shape[0] == tar.shape[0], ('inp.shape: %s tar.shape: %s' % (inp.shape, tar.shape))
        self._n_dat = inp.shape[0]
        self._inp = inp
        self._tar = tar
        self._epochs_completed = 0
        self._epoch_i = 0

    @property
    def inp(self):
        return self._inp

    @property
    def tar(self):
        return self._tar

    @property
    def num_datasets(self):
        return self._n_dat

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        
        start = self._epoch_i
        
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
          perm0 = np.arange(self._n_dat)
          np.random.shuffle(perm0)
          self._inp = self.inp[perm0]
          self._tar = self.tar[perm0]
        
        # Go to the next epoch
        if start + batch_size > self._n_dat:
            
            # Finished epoch
            self._epochs_completed += 1
            
            # Get the rest examples in this epoch
            rest_num_datasets = self._n_dat - start
            inputs_rest_part = self._inp[start:self._n_dat]
            targets_rest_part = self._tar[start:self._n_dat]
            
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._n_dat)
                np.random.shuffle(perm)
                self._inp = self.inp[perm]
                self._tar = self.tar[perm]
            
            # Start next epoch
            start = 0
            self._epoch_i = batch_size - rest_num_datasets
            end = self._epoch_i
            inputs_new_part = self._inp[start:end]
            targets_new_part = self._tar[start:end]
            
            return np.concatenate((inputs_rest_part, inputs_new_part), axis=0), np.concatenate((targets_rest_part, targets_new_part), axis=0)
        
        else:
            
            self._epoch_i += batch_size
            end = self._epoch_i
            
            return self._inp[start:end], self._tar[start:end]

# Read the data 
def load_data(n, dir):

    # Reading data 
    q = np.loadtxt(dir + 'data_q_30.dat', dtype='float64')
    t = np.loadtxt(dir + 'data_t_30.dat', dtype='float64')

    # Dataset size
    n_cases = int(q.size / (n*n))

    # Reshape data
    q = np.reshape(q, (n_cases, n, n))
    t = np.reshape(t, (n_cases, n, n))

    # Find min and max for normalization (can be avoided)
    q_max = np.max(q)
    q_min = np.min(q)
    t_max = np.max(t)
    t_min = np.min(t)

    # Normalize target data in [0, 1]; inputs are normalized just before NN
    q = (q - q_min) / (q_max - q_min)

    # Split dataset on training, validation, and test
    num_test = 5
    test_cases = [1, 12, 16, 23, 28]
    t_test = np.zeros(shape=(num_test, n, n), dtype='float64')
    q_test = np.zeros(shape=(num_test, n, n), dtype='float64')

    num_valid = 5
    valid_cases = [3, 7, 15, 21, 27]
    t_valid = np.zeros(shape=(num_valid, n, n), dtype='float64')
    q_valid = np.zeros(shape=(num_valid, n, n), dtype='float64')

    num_train = n_cases - num_test - num_valid
    train_cases = [0, 2, 4, 5, 6, 8, 9, 10, 11, 13, 14, 17, 18, 19, 20, 22, 24, 25, 26, 29]
    t_train = np.zeros(shape=(num_train, n, n), dtype='float64')
    q_train = np.zeros(shape=(num_train, n, n), dtype='float64')
    
    i = 0
    j = 0
    l = 0
    for k in range(0,n_cases):

        if (i < num_valid):
            if (k == valid_cases[i]):
                t_valid[i, :, :] = t[k, :, :]
                q_valid[i, :, :] = q[k, :, :]
                i += 1
        
        if (j < num_test):
            if (k == test_cases[j]):
                t_test[j, :, :] = t[k, :, :]
                q_test[j, :, :] = q[k, :, :]
                j += 1

        if (l < num_train):
            if (k == train_cases[l]):
                t_train[l, :, :] = t[k, :, :]
                q_train[l, :, :] = q[k, :, :]
                l += 1

    # Split training and validation fields [NxN] in mini-batches [3x3]
    num_train_split = (n-2)*(n-2)*num_train
    t_train_split = np.zeros(shape=(num_train_split, 3, 3), dtype='float64')
    q_train_split = np.zeros(shape=(num_train_split, 3, 3), dtype='float64')

    num_valid_split = (n-2)*(n-2)*num_valid
    t_valid_split = np.zeros(shape=(num_valid_split, 3, 3), dtype='float64')
    q_valid_split = np.zeros(shape=(num_valid_split, 3, 3), dtype='float64')

    l = 0
    for k in range(0,num_train):
        for i in range(0,n-2):      
            for j in range(0,n-2):
                t_train_split[l, :, :] = t_train[k, i:(i+3), j:(j+3)]
                q_train_split[l, :, :] = q_train[k, i:(i+3), j:(j+3)]
                l += 1

    l = 0
    for k in range(0,num_valid):
        for i in range(0,n-2):
            for j in range(0,n-2):
                t_valid_split[l, :, :] = t_valid[k, i:(i+3), j:(j+3)]
                q_valid_split[l, :, :] = q_valid[k, i:(i+3), j:(j+3)]
                l += 1
    
    # Create datasets 
    train = DataSet(t_train_split, q_train_split)
    valid = DataSet(t_valid_split, q_valid_split)
    test = DataSet(t_test, q_test)
    data = base.Datasets(train=train, validation=valid, test=test)
    
    return data, q_max, q_min, t_max, t_min