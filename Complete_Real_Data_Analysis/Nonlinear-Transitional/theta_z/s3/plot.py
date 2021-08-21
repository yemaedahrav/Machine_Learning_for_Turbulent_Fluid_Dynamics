import sys
import os
import time
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from pylab import savefig

N_t = 55
N_th = 128
N_z = 512
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'adjusted_heatmaps/')

for time_stamp in range(N_t):
    print(time_stamp)
    eig = np.loadtxt("s3_eigen_"+str(time_stamp)+".txt", dtype=np.float64)
    eig = np.asarray(eig)
    eig = np.reshape(eig, (N_th, N_z))
    eig = np.clip(eig, 1e-8 ,1e7)

    f = plt.figure()
    f.set_figwidth(12)
    f.set_figheight(10)

    # plt.plot(eig,color='teal', marker='*', linestyle='dotted',linewidth=0.8, markersize=3)
    # plt.xlabel('Wavenumebr m at time t={}'.format(time_stamp))
    # plt.ylabel('Eigenvalue')
    
    # svm = sns.heatmap(eig, cmap='RdBu_r', cbar=True, square=True, xticklabels="auto", yticklabels="auto", norm=LogNorm())
    plt.pcolormesh(eig, cmap='RdBu_r', norm=LogNorm())
    plt.xlabel('z')
    plt.ylabel('$\theta$')
    plt.title("Eigenvalues at t = {}".format(time_stamp))
    plt.colorbar(orientation='horizontal')
    file_name = "s3_heatmap" + str(time_stamp)+"_adjusted.png"
    plt.savefig(results_dir + file_name)
    plt.close()