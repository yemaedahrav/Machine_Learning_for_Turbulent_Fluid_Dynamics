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

file_base = "ret60_s1_z-theta_t_"
results_base = "/home/ameyv6/Machine_Learning_for_Turbulent_Fluid_Dynamics/Complete_Real_Data_Analysis/Nonlinear-Transitional/ret60_3_nl_g_s1"
colors = ['maroon','orange','lime','cyan','navy']

file_list = []
for it in range(N_t):
    cur_file = file_base+str(it)+".txt"
    file_list.append(cur_file)

array_dict = {}
for i in range(0,len(file_list),17):
    print(i)
    eig = np.loadtxt(file_list[i], dtype=np.float64)
    eig = np.asarray(eig)
    eig = np.reshape(eig, (N_th, N_z))
    eig = np.clip(eig, 1e-8 ,1e6)
    array_dict[i] = eig       
    
for th in range(0,N_th,15):
    print("th: ", th)
    fig = plt.figure()
    fig.set_figwidth(15)
    fig.set_figheight(8)

    for iter in range(5):   
        plt.plot(array_dict[17*iter][:128,th],color=colors[iter], marker='o', linestyle='dashed',linewidth=1, markersize=4, label=iter)
  
    plt.xlabel('Varying kz at fixed wavenumber m = {}'.format(th))
    plt.ylabel('Eigenvalue')
    plt.yscale('log')
    plt.legend()
    plt.title("Eigenvalues at fixed m = {}".format(th))

    file_name = "m_" + str(th)+ "_time_series.png"
    plt.savefig(results_base + "/time_varying/" +file_name)
    plt.close()

for z in range(0,N_z,15):
    print("z: ", z)
    fig = plt.figure()
    fig.set_figwidth(15)
    fig.set_figheight(8)

    for iter in range(5):   
        plt.plot(array_dict[17*iter][z,:128],color=colors[iter], marker='o', linestyle='dashed',linewidth=1, markersize=4, label=iter)
  
    plt.xlabel('Varying m at fixed wavenumber kz = {}'.format(z))
    plt.ylabel('Eigenvalue')
    plt.yscale('log')
    plt.legend()
    plt.title("Eigenvalues at fixed kz = {}".format(z))

    file_name = "kz_" + str(z)+ "_time_series.png"
    plt.savefig(results_base + "/time_varying/" +file_name)
    plt.close()