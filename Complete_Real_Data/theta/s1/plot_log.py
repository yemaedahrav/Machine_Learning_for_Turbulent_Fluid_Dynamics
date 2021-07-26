import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

N_t = 100
script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'logarithmic/')

for time_stamp in range(N_t):
    print(time_stamp)
    eig = np.loadtxt(str(time_stamp)+".txt", dtype=np.float64)
    eig = np.asarray(eig)

    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(7)

    plt.plot(eig,color='teal', marker='*', linestyle='dotted',linewidth=0.8, markersize=3)
    plt.xlabel('Wavenumebr m at time t={}'.format(time_stamp))
    plt.ylabel('Eigenvalue')
    plt.yscale('log')
    file_name = str(time_stamp)+"_log.png"
    plt.savefig(results_dir + file_name, dpi=200)