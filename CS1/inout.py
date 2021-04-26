# Third-party libraries
import os
import numpy as np

# Set paths to folders
def set_paths():

    # Path to the output folder
    work_dir = '../work/'
    if not os.path.exists(work_dir):
        sys.exit('WORKING FOLDER DOES NOT EXIST!')

    # Path to the data directory
    data_dir = work_dir + 'data/'
    if not os.path.exists(data_dir):
        sys.exit('FOLDER WITH DATA DOES NOT EXIST!')

    # Path to the figures directory
    fig_dir = work_dir + 'fig/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        os.makedirs(fig_dir + 'train/')
        os.makedirs(fig_dir + 'train/split/')
        os.makedirs(fig_dir + 'valid/')
        os.makedirs(fig_dir + 'valid/split/')
        os.makedirs(fig_dir + 'test/')
        os.makedirs(csv_dir + 'test/init/')
        print('FOLDERS FOR FIGURES WERE CREATED!')

    # Path to .csv files directory
    csv_dir = work_dir + 'csv/'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
        os.makedirs(csv_dir + 'train/')
        os.makedirs(csv_dir + 'valid/')
        os.makedirs(csv_dir + 'test/')
        print('FOLDERS FOR .CSV FILES WERE CREATED!')

    # Path to the tecplot directory
    tec_dir = work_dir + 'tec/'
    if not os.path.exists(tec_dir):
        os.makedirs(tec_dir)
        os.makedirs(tec_dir + 'train/')
        os.makedirs(tec_dir + 'valid/')
        os.makedirs(tec_dir + 'test/')
        print('FOLDERS FOR TECPLOT FILES WERE CREATED!')

    # Path to the saved model directory
    nn_dir = work_dir + 'nn/'
    if not os.path.exists(nn_dir):
        os.makedirs(nn_dir)
        print('FOLDER FOR NN FILES WAS CREATED!')

    return work_dir, data_dir, fig_dir, csv_dir, tec_dir, nn_dir

# Fields output for tecplot
def write_tec_var(n, var1, var2, x, y, time, res_plt, var_name1, var_name2):
    
    var1 = np.transpose(var1)
    var2 = np.transpose(var2)
    res_plt.write('title="fields" variables="x" "y" "' + var_name1 + '"' + ' "' + var_name2 + '" \n')
    res_plt.write('zone t="time=' + str(time) + '" i=' + str(n) + ' j=' + str(n) + ' f=block')

    # x grid
    for i in range(0,n):
        res_plt.write("\n")
        for j in range(0,n):
            res_plt.write("%.12f" % x[i] + " ")
    # y grid
    for i in range(0,n):
        res_plt.write("\n")
        for j in range(0,n):
            res_plt.write("%.12f" % y[j] + " ")
    # var
    for i in range(0,n):
        res_plt.write("\n")
        for j in range(0,n):
            res_plt.write("%.12f" % var1[i, j] + " ")
    for i in range(0,n):
        res_plt.write("\n")
        for j in range(0,n):
            res_plt.write("%.12f" % var2[i, j] + " ")
    res_plt.write("\n")

