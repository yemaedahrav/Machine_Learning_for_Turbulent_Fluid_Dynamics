import sys
import time
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
from numba import jit,njit

start_time = time.time()

f = h5py.File('/users/avarhard/data/shared_data/ret60_3_nl_g/ret60_3_nl_g_s4.h5', 'r')

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
def compute_eigen(r, z, N_th, N_r, N_z, v_r_f, v_theta_f, v_z_f, p_f):
    dz = z[1]-z[0]
    Q = np.zeros((N_th,N_z),dtype=np.float64)    

    for m in range(N_th):
        #print("Wavenumber azimuthal: ",m)
        #print(type(eigen_val))
        for k in range(N_z):
            #print("Wavenumber axial: ",k)
            eigen_val = 0.0
            for R in range(N_r):
                #print("R: ",R)
                if R==0:
                    rdr = r[R]*(r[1]-r[0]) 
                else:
                    rdr = r[R]*(r[R]-r[R-1])

                Q_v_r = v_r_f[k][m][R]
                Q_v_theta = v_theta_f[k][m][R]
                Q_v_z = v_z_f[k][m][R]
                Q_p = p_f[k][m][R]

                power_amplitude = (Q_v_r)*np.conj(Q_v_r) + (Q_v_theta)*np.conj(Q_v_theta) + (Q_v_z)*np.conj(Q_v_z) + (Q_p)*np.conj(Q_p)
                # print(type(power_amplitude))
                power_amplitude = power_amplitude.real
                # print(power_amplitude)
                # print(power_amplitude.shape
                eigen_val += (rdr*power_amplitude)
            Q[m][k] = 2*(np.pi)*eigen_val
    return Q

for time_stamp in range(N_t):
    print("\nTimeStamp: ", time_stamp, "\n")    

    u = f['tasks']['u'][time_stamp]
    v = f['tasks']['v'][time_stamp]
    w = f['tasks']['w'][time_stamp]
    p = f['tasks']['p'][time_stamp]    

    v_r = np.asarray(u)
    v_theta = np.asarray(v)
    v_z = np.asarray(w)
    p = np.asarray(p)

    v_r_f = np.fft.fft2(v_r, axes=(0,1))
    v_theta_f = np.fft.fft2(v_theta, axes=(0,1))
    v_z_f = np.fft.fft2(v_z, axes=(0,1))
    p_f = np.fft.fft2(p, axes=(0,1))
    
    Q_t = np.zeros((N_th,N_z),dtype=np.float64)
    Q_t = compute_eigen(r, z, N_th, N_r, N_z, v_r_f, v_theta_f, v_z_f, p_f)
    np.savetxt("/users/avarhard/scratch/ret60_3_nl_g_s4/"+"ret60_s4_z-theta_t_"+str(time_stamp)+".txt", Q_t)
    print("Q_t: ",Q_t)
    print("Q Shape: ",Q_t.shape)
    print("\n")

f.close()
end_time = time.time()
print('Time Duration: {}'.format(end_time - start_time))