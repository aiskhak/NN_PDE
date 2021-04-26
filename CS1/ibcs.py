####################################################
# I&BCs, GRID GENERATION ###########################
####################################################

# Third-party libraries
import numpy as np

# Set I&BCs
def set_ibcs(n, h, q_data, t_data, n_cases, is_data_ibc, q_max, q_min):

    tw = 1200.0     # West temperature [K]
    te_0 = 800.0    # East temperature for training dataset No. 1 [K]
    te_st = 5.0     # Step for other training datasets [K]

    # east temperature
    te = np.zeros([n_cases], dtype=np.float64)
    for k in range(n_cases):
        te[k] = te_0 + te_st*k

    # use data to set I&BCs or zeros        
    if is_data_ibc:
        q_init = q_data*(q_max - q_min) + q_min
        t_init = t_data

    # set linear distribution
    else:

        # arrays
        q_init = np.zeros([n_cases, n, n], dtype=np.float64)
        t_init = np.zeros([n_cases, n, n], dtype=np.float64)

        # linear ICs
        for k in range(n_cases):
            for j in range(n):
                for i in range(n):
                    t_init[k, j, i] = tw - (tw - te[k])*j*h
                    q_init[k, j, i] = -4.e2*np.sqrt(t_init[k,j,i]) + 1.e1*(t_init[k,j,i] - 1.e2) + 1.e-8*np.power(t_init[k,j,i], 4)
    
    # Separeted sources
    q_1 = np.zeros([n_cases, n, n], dtype=np.float64)
    q_2 = np.zeros([n_cases, n, n], dtype=np.float64)
    q_3 = np.zeros([n_cases, n, n], dtype=np.float64)
    for k in range(n_cases):
        for j in range(n):
            for i in range(n):
                q_1[k, j, i] = -4.e2*np.sqrt(t_init[k, j, i])
                q_2[k, j, i] = 1.e1*(t_init[k, j, i] - 1.e2)
                q_3[k, j, i] = 1.e-8*np.power(t_init[k,j,i], 4)
                    
    return q_init, t_init, q_1, q_2, q_3

# grid generation
def grid_gen(n, l):

    h = l/(n - 1)
    x, y = np.meshgrid(np.arange(0, n), np.arange(0, n))        
    x = x[0, :]*h
    y = y[:, 0]*h

    x_split = np.zeros(shape=(n-2,3), dtype='float64')
    y_split = np.zeros(shape=(n-2,3), dtype='float64')
    for k in range(0,n-2):
        x_split[k,:] = x[k:(k+3)]
        y_split[k,:] = y[k:(k+3)]

    return h, x_split, y_split, x, y



