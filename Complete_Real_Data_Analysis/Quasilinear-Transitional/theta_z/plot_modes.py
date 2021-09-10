import sys
import os
import time
import numpy as np  
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm

font = {'size'   : 15}
matplotlib.rc('font', **font)

N_r = 96
N_th = 128
N_z = 512
N_t = 100

file_base = "/home/ameyv6/Machine_Learning_for_Turbulent_Fluid_Dynamics/Complete_Real_Data_Analysis/Quasilinear-Transitional/theta_z/s1/"
results_base = file_base+"combined-modes/"
colors = ['brown','gray','pink','cyan','olive']

file_list = []
for it in range(N_t):
    cur_file = file_base+"s-1_t-"+str(it)+".txt"
    file_list.append(cur_file)

for i in range(len(file_list)):
    print(i)
    eig = np.loadtxt(file_list[i], dtype=np.float64)
    eig = np.asarray(eig)
    eig = np.reshape(eig, (N_th, N_z))
    eig = np.clip(eig, 1e-8 ,1e6)     
    
    fig = plt.figure()
    fig.set_figwidth(15)
    fig.set_figheight(8)

    for th in range(5):   
        plt.plot(eig[th*31,:],color=colors[th], marker='o', linestyle='dashed',linewidth=1, markersize=4, label="m = "+str(31*th))

    plt.xlabel(r'$k_Z$')
    plt.ylabel('Eigenvalue')
    plt.yscale('log')
    plt.legend(fontsize = 'medium', title_fontsize = 'medium')
    plt.title("Eigenvalues at fixed m values (QL Transitional)")

    file_name = "ql_s1_t_"+str(i)+"_fixed_m.png"
    plt.savefig(results_base+file_name)
    plt.close()

    fig = plt.figure()
    fig.set_figwidth(15)
    fig.set_figheight(8)

    for z in range(5):   
        plt.plot(eig[:,z*127],color=colors[z], marker='o', linestyle='dashed',linewidth=1, markersize=4, label=r'$k_z$'+" = "+str(127*z))

    plt.xlabel('m')
    plt.ylabel('Eigenvalue')
    plt.yscale('log')
    plt.legend(fontsize = 'medium', title_fontsize = 'medium')
    plt.title("Eigenvalues at fixed "+r'$k_z$'+ " values (QL Transitional)")

    file_name = "ql_s1_t_"+str(i)+"_fixed_kz.png"
    plt.savefig(results_base+file_name)
    plt.close()