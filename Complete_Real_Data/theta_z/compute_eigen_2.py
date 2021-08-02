import sys
import time
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
import seaborn as sns
from numba import jit,njit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from pylab import savefig

start_time = time.time()

f = h5py.File('state_variables24_10_g_s3.h5', 'r')

r = f['scales']['r']['1.0']
z = f['scales']['z']['1.0']
th = f['scales']['th']['1.0']
t = f['scales']['sim_time']

th = np.asarray(th)
z = np.asarray(z)
r = np.asarray(r)
t = np.asarray(t)

N_r = r.shape[0]
N_th = th.shape[0]
N_z = z.shape[0]
N_t = t.shape[0]

@njit(fastmath = True, parallel = True)
def compute_eigen(r, z, N_th, N_r, N_z, v_r_f, v_theta_f, v_z_f):
    dz = z[1]-z[0]
#     kTh = np.arange(N_th, dtype=np.int32)
#     kTh = np.reshape(kTh,(kTh.shape[0],1))
    Q = np.zeros((N_th,N_z),dtype=np.float64)    
#     print(Q.shape)
#     print("Q: ",type(Q))
#     print("Q[0]: ",type(Q[0][0]))

    for m in range(N_th):
        print("Wavenumber azimuthal: ",m)
        #print(type(eigen_val))
        for k in range(N_z):
            print("Wavenumber axial: ",k)
            eigen_val = 0.0
            for R in range(N_r):
                # print("R: ",R)
                if R==0:
                    rdr = r[R]*(r[1]-r[0]) 
                else:
                    rdr = r[R]*(r[R]-r[R-1])

                Q_v_r = v_r_f[k][m][R]
                Q_v_theta = v_theta_f[k][m][R]
                Q_v_z = v_z_f[k][m][R]

                power_amplitude = (Q_v_r)*np.conj(Q_v_r) + (Q_v_theta)*np.conj(Q_v_theta) + (Q_v_z)*np.conj(Q_v_z)
                # print(type(power_amplitude))
                power_amplitude = power_amplitude.real
                # print(power_amplitude)
                # print(power_amplitude.shape
                eigen_val += (rdr*power_amplitude)
            Q[m][k] = 2*(np.pi)*eigen_val
 
#     par_func = np.vectorize(compute_mode)    
#     Q = par_func(kTh);
#     print(Q.shape)
#     print("Q: ",type(Q))
    return Q

for time_stamp in range(N_t):
    print("\nTimeStamp: ", time_stamp, "\n")    

    ul = f['tasks']['ul'][time_stamp]
    uh = f['tasks']['uh'][time_stamp]

    vl = f['tasks']['vl'][time_stamp]
    vh = f['tasks']['vh'][time_stamp]

    wl = f['tasks']['wl'][time_stamp]
    wh = f['tasks']['wh'][time_stamp]

    ul = np.asarray(ul)
    uh = np.asarray(uh)

    vl = np.asarray(vl)
    vh = np.asarray(vh)

    wl = np.asarray(wl)
    wh = np.asarray(wh)

    v_r = ul + uh
    v_theta = vl + vh
    v_z = wl + wh

    # print("v_r shape: ",v_r.shape)
    # print("v_theta shape: ",v_theta.shape)
    # print("v_z shape: ",v_z.shape)

    v_r_f = np.fft.fft2(v_r, axes=(0,1))
    v_theta_f = np.fft.fft2(v_theta, axes=(0,1))
    v_z_f = np.fft.fft2(v_z, axes=(0,1))
    
    Q_t = np.zeros((N_th,N_z),dtype=np.float64)
    Q_t = compute_eigen(r, z, N_th, N_r, N_z, v_r_f, v_theta_f, v_z_f)
    np.savetxt("s3_eigen_"+str(time_stamp)+".txt", Q_t)
    print("Q_t: ",Q_t)
    print("Q Shape: ",Q_t.shape)
    
    fig = plt.figure()
    fig.set_figwidth(12)
    fig.set_figheight(10)
    
    svm = sns.heatmap(Q_t, cbar=True, square=True, xticklabels="auto", yticklabels="auto", norm=LogNorm())
#     fig = svm.get_figure()    
#     fig.savefig(str(time_stamp)+'.png', dpi=400)
    file_name = "s3_log_heatmap_"+str(time_stamp)+".png"
    plt.savefig(file_name, facecolor='white')
    # plt.show()

#     fig = plt.figure()
    
#     ax = plt.axes(projection ='3d')
#     ax = Axes3D(fig)
                
#     ax.set_xlabel(r'$\theta$', fontsize=8)
#     ax.set_ylabel(r'$z$', fontsize=8)
#     ax.set_zlabel(r'$\lambda$', fontsize=10)

#     surf = ax.plot_surface(th, z, Q_t, cmap='viridis')
#     file_name = str(time_stamp) +".png"
#     plt.savefig(file_name)
#     plt.show()

f.close()
end_time = time.time()
print('Time Duration: {}'.format(end_time - start_time))