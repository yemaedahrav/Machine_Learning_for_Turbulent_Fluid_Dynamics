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

for time_stamp in range(43,54):
    print(time_stamp)
    dir_name = "time_stamp_" + str(time_stamp)+"/"
    results_dir = os.path.join(script_dir, dir_name)
    eig = np.loadtxt("s3_eigen_"+str(time_stamp)+".txt", dtype=np.float64)
    eig = np.asarray(eig)
    eig = np.reshape(eig, (N_th, N_z))
    eig = np.clip(eig, 1e-8, 1e7)
    print("File Read")
    for th in range(N_th):
        print("th: ", th)
        Q_th = eig[th, :]
        fig = plt.figure()
        fig.set_figwidth(15)
        fig.set_figheight(8)

        plt.plot(Q_th,color='green', marker='o', linestyle='dashed',linewidth=1, markersize=5)
        plt.xlabel('Wavenumebr m at time t={}, m = {}'.format(time_stamp, th))
        plt.ylabel('Eigenvalue')
        plt.yscale('log')
        plt.title("Eigenvalues at fixed t = {}, m = {}".format(time_stamp, th))

        file_name = "m_" + str(th)+ "_time_" +str(time_stamp)+".png"
        plt.savefig(results_dir + file_name)
        plt.close()
        #plt.show()

    for z in range(N_z):
        print("z: ", z)
        Q_z = eig[:, z]
        fig = plt.figure()
        fig.set_figwidth(15)
        fig.set_figheight(8)

        plt.plot(Q_z,color='green', marker='o', linestyle='dashed',linewidth=1, markersize=5)
        plt.xlabel('Wavenumebr m at time t={}, kz = {}'.format(time_stamp, z))
        plt.ylabel('Eigenvalue')
        plt.yscale('log')
        plt.title("Eigenvalues at fixed t = {}, kz = {}".format(time_stamp, z))

        file_name = "kz_" + str(z)+ "_time_" +str(time_stamp)+".png"
        plt.savefig(results_dir + file_name)
        plt.close()