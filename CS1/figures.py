####################################################
# MODULE FOR FIGURES ###############################
####################################################

# Third-party libraries
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import PIL

# GRAPHICAL PARAMETERS
rcParams.update({'figure.max_open_warning': 0})
rcParams['font.family'] = 'Times New Roman'
colorInterpolation = 30
colorMap = plt.cm.jet
set_dpi = 300
size_12 = 12
rcParams['font.size'] = size_12
rcParams['mathtext.default'] = 'regular'
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

# FIELDS: TEMPERATURE AND SOURCES    
def fields_fig_out(k, x, y, t, q, data_dir):

    # Temperature
    fig = plt.figure(figsize=(2.7, 2.2))
    plt.contourf(x, y, t, colorInterpolation, cmap=colorMap)
    #plt.grid(color='r', linestyle='-', linewidth=2)
    plt.colorbar()
    fig.savefig(data_dir + 't_case_' + str(k+1) + '.tif', dpi=set_dpi, bbox_inches='tight')

    # Sources
    fig = plt.figure(figsize=(2.7, 2.2))
    plt.contourf(x, y, q, colorInterpolation, cmap=colorMap)
    plt.colorbar()
    fig.savefig(data_dir + 'q_case_' + str(k+1) + '.tif', dpi=set_dpi, bbox_inches='tight')

# COST
def cost_fig_out(epochs, plot_1, plot_1_valid, plot_2, plot_2_valid, data_dir):

    fig = plt.figure(figsize=(10.0, 2.2))
    ax = plt.subplot(1, 2, 1)
    plt.yscale("log")
    plt.plot(np.arange(0, int(epochs), 1), plot_1[:epochs], color='black', linewidth=1, label='Training')
    plt.plot(np.arange(0, int(epochs), 1), plot_1_valid[:epochs], color='red', linewidth=1, label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Cost, K')
    ax = plt.subplot(1, 2, 2)
    plt.yscale("log")
    plt.plot(np.arange(0, int(epochs), 1), plot_2[:epochs], color='black', linewidth=1, label='Training')
    plt.plot(np.arange(0, int(epochs), 1), plot_2_valid[:epochs], color='red', linewidth=1, label='Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('L2(q), K/m$^{2}$')
    fig.savefig(data_dir + 'cost.tif', dpi=set_dpi, bbox_inches='tight')

# SOURCES
def q_fig_out(k, x, y, q_nn, q_targ, data_dir):

    fig = plt.figure(figsize=(3.4, 9.0))
    ax = plt.subplot(3, 1, 1)
    plt.contourf(x, y, q_nn, colorInterpolation, cmap=colorMap)
    plt.colorbar()
    #fig.savefig(data_dir + 'nn_case_' + str(k+1) + '_q.tif', dpi=set_dpi, bbox_inches='tight')
    plt.title("NN prediction", fontsize=size_12)
    #fig = plt.figure(figsize=(2.7, 2.2))
    ax = plt.subplot(3, 1, 2)
    plt.contourf(x, y, q_targ, colorInterpolation, cmap=colorMap)
    plt.colorbar()
    #fig.savefig(data_dir + 'targ_case_' + str(k+1) + '_q.tif', dpi=set_dpi, bbox_inches='tight')
    plt.title("Exact", fontsize=size_12)
    #fig = plt.figure(figsize=(2.7, 2.2))
    ax = plt.subplot(3, 1, 3)
    plt.contourf(x, y, q_targ-q_nn, colorInterpolation, cmap=colorMap)
    plt.colorbar()
    plt.title("Difference", fontsize=size_12)
    fig.savefig(data_dir + 'case_' + str(k+1) + '_q.tif', dpi=set_dpi, bbox_inches='tight')

# DEVIATIONS: TEMPERATURE
def dev_fig_out(t, t_data, data_dir):

    fig_1D_ = plt.figure(figsize=(2.2, 2.2))
    plt.plot(t_data, t_data, color='black', linewidth=1)
    #plt.plot(t, t_data, 'ro', label='NN solution', markersize=1)
    plt.plot(t, t_data, 'ro',  markersize=1)
    #plt.legend(frameon=False, handletextpad=0.0, loc='upper left')
    plt.xlabel('T, K', fontsize=size_12)
    plt.ylabel('T, K', fontsize=size_12)
    fig_1D_.savefig(data_dir + 'dev_t.tif', dpi=set_dpi, bbox_inches='tight')

# 1D RESULTS: VELOCITY AND PRESSURE CONVERGENCE, VELOCITY PROFILES
def one_dim_fig_out(k, it, abs_er_max, eps, data_dir):

    # Convergence
    fig = plt.figure(figsize=(2.2, 2.2))
    ax = plt.subplot(1, 1, 1)
    ax.plot(np.arange(0, int(it), 1), abs_er_max[:it, k], 'o', c='red')
    ax.set_title('Convergence for case ' + str(k+1), fontsize=size_12)
    ax.set_yscale('log')
    plt.xlabel('Iteration', fontsize=size_12)
    plt.ylabel('Max Grid Residual', fontsize=size_12)
    ax.grid(which='minor', alpha=0.1)
    ax.grid(which='major', alpha=0.5)
    min = eps*1.e-1
    plt.ylim(bottom=min)
    fig.savefig(data_dir + 'case_' + str(k+1) + '_1D.png', format='png', dpi=set_dpi, bbox_inches='tight')

# DEVIATIONS: SOURCE
def q_dev_fig_out(k, q, q_data, data_dir):

    fig_1D_ = plt.figure(figsize=(2.2, 2.2))
    plt.plot(q_data, q_data, color='black', linewidth=1)
    #plt.plot(q, q_data, 'ro', label='NN', markersize=2)
    plt.plot(q, q_data, 'ro', markersize=1)
    plt.xlabel(r'q, K/n$^{2}$')
    plt.ylabel(r'q, K/n$^{2}$')
    #plt.legend(frameon=False, handletextpad=0.0)
    fig_1D_.savefig(data_dir + 'case_' + str(k+1) + '_q_dev.tif', dpi=set_dpi, bbox_inches='tight')

# Separated sources
def sep_q_fig_out(k, x, y, q1, q2, q3, dir):

    fig = plt.figure(figsize=(12, 2.8))
    ax = plt.subplot(1, 3, 1)
    plt.contourf(x, y, q1, colorInterpolation, cmap=colorMap)
    plt.colorbar()

    ax = plt.subplot(1, 3, 2)
    plt.contourf(x, y, q2, colorInterpolation, cmap=colorMap)
    plt.colorbar()

    ax = plt.subplot(1, 3, 3)
    plt.contourf(x, y, q3, colorInterpolation, cmap=colorMap)
    plt.colorbar()

    fig.savefig(dir + 'q_case_' + str(k+1) + '.tif', dpi=set_dpi, bbox_inches='tight')