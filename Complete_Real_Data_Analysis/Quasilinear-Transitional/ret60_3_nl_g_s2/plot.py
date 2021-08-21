import sys
import os
import time
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

N_r = 96
N_th = 256
N_z = 256
N_t = 69

file_base = "ret60_s2_z-theta_t_"

file_list = []
for it in range(N_t):
    cur_file = file_base+str(it)+".txt"
    file_list.append(cur_file)

for i in range(N_t):
    print(i)
    eig = np.loadtxt(file_list[i], dtype=np.float64)
    eig = np.asarray(eig)
    eig = np.reshape(eig, (N_th, N_z))
    eig = np.clip(eig, 1e-8 ,1e7)
    
    result_dir = "/users/avarhard/scratch/ret60_3_nl_g_s2/t_"+str(i)+"/"
    f = plt.figure()
    f.set_figwidth(12)
    f.set_figheight(10)
 
    plt.pcolormesh(eig, cmap='viridis', norm=LogNorm())
    plt.xlabel('z')
    plt.ylabel(r'$\theta$')
    plt.title("Eigenvalues at t = {}".format(i))
    plt.colorbar(orientation='horizontal')
    file_name = "s2_heatmap_" + str(i)+".png"
    plt.savefig(result_dir + file_name)
    plt.close()

    for th in range(0,N_th,16):
        print("th: ", th)
        Q_th = eig[th, :]
        fig = plt.figure()
        fig.set_figwidth(15)
        fig.set_figheight(8)

        plt.plot(Q_th,color='green', marker='o', linestyle='dashed',linewidth=1, markersize=5)
        plt.xlabel('Wavenumebr m at time t={}, m = {}'.format(i, th))
        plt.ylabel('Eigenvalue')
        plt.yscale('log')
        plt.title("Eigenvalues at fixed t = {}, m = {}".format(i, th))

        file_name = "m_" + str(th)+ "_time_" +str(i)+".png"
        plt.savefig(result_dir + file_name)
        plt.close()

    for z in range(0,N_z,16):
        print("z: ", z)
        Q_z = eig[:, z]
        fig = plt.figure()
        fig.set_figwidth(15)
        fig.set_figheight(8)

        plt.plot(Q_z,color='green', marker='o', linestyle='dashed',linewidth=1, markersize=5)
        plt.xlabel('Wavenumebr m at time t={}, kz = {}'.format(i, z))
        plt.ylabel('Eigenvalue')
        plt.yscale('log')
        plt.title("Eigenvalues at fixed t = {}, kz = {}".format(i, z))

        file_name = "kz_" + str(z)+ "_time_" +str(i)+".png"
        plt.savefig(result_dir + file_name)
        plt.close()
