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
colors = ['maroon','orange','lime','deepskyblue','navy']

file_list = []
for it in range(N_t):
    cur_file = file_base+str(it)+".txt"
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
    
    plt.plot([], [], ' ', label="m")
    for th in range(5):   
        plt.plot(eig[th*31,:128],color=colors[th], marker='o', linestyle='dashed',linewidth=1, markersize=4, label=31*th)

    plt.xlabel(r"$k_z$")
    plt.ylabel('Eigenvalue')
    plt.yscale('log')
    plt.legend()
    plt.title("Eigenvalues at fixed m values")

    file_name = "t_"+str(i)+"_fixed_m_adjusted.png"
    plt.savefig(results_base + "/combined_plots/" +file_name)
    plt.close()

    fig = plt.figure()
    fig.set_figwidth(15)
    fig.set_figheight(8)
    
    plt.plot([], [], ' ', label=r"$k_z$")
    for z in range(5):   
        plt.plot(eig[:128,z*31],color=colors[z], marker='o', linestyle='dashed',linewidth=1, markersize=4, label=31*z)

    plt.xlabel('m')
    plt.ylabel('Eigenvalue')
    plt.yscale('log')
    plt.legend()
    plt.title(r"Eigenvalues at fixed $k_z$ values")

    file_name = "t_"+str(i)+"_fixed_kz_adjusted.png"
    plt.savefig(results_base + "/combined_plots/" +file_name)
    plt.close()