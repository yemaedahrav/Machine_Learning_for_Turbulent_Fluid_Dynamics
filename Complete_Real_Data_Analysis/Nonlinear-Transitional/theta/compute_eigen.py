import sys
import time
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py
from numba import jit,njit

start_time = time.time()

f = h5py.File('state_variables24_10_g_s1.h5', 'r')

def printname(name):
    print(name)
f.visit(printname)

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

print("N_r: ", N_r)
print("N_th: ", N_th)
print("N_z: ", N_z)
print("N_t: ", N_t)


# Q = np.zeros((N_th,N_t), dtype = np.float64)

@njit(fastmath = True, parallel = True)
def compute_eigen(r, z, N_th, N_r, N_z, v_r_f, v_theta_f, v_z_f):
    dz = z[1]-z[0]
#     kTh = np.arange(N_th, dtype=np.int32)
#     kTh = np.reshape(kTh,(kTh.shape[0],1))
    Q = np.zeros((N_th,1),dtype=np.float64)    
#     print(Q.shape)
#     print("Q: ",type(Q))
#     print("Q[0]: ",type(Q[0][0]))

    for m in range(N_th):
        print("Wavenumber: ",m)
        eigen_val = 0.0
        #print(type(eigen_val))
        for R in range(N_r):
            print("R: ",R)
            if R==0:
                rdr = r[R]*(r[1]-r[0]) 
            else:
                rdr = r[R]*(r[R]-r[R-1])
            #   rdr = r[R]*(r[R]-r[R-1])
            for Z in range(1,N_z):
                Q_v_r = v_r_f[Z][m][R]
                Q_v_theta = v_theta_f[Z][m][R]
                Q_v_z = v_z_f[Z][m][R]
                
                power_amplitude = (Q_v_r)*np.conj(Q_v_r) + (Q_v_theta)*np.conj(Q_v_theta) + (Q_v_z)*np.conj(Q_v_z)
                # print(type(power_amplitude))
                power_amplitude = power_amplitude.real
                # print(power_amplitude)
                # print(power_amplitude.shape
                eigen_val += (rdr*power_amplitude)
        Q[m][0] = 2*(np.pi)*dz*eigen_val

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

    v_r_f = np.fft.fft(v_r, axis=1)
    v_theta_f = np.fft.fft(v_theta, axis=1)
    v_z_f = np.fft.fft(v_z, axis=1)
    
    Q_t = np.zeros((N_th,1),dtype=np.float64)
    Q_t = compute_eigen(r, z, N_th, N_r, N_z, v_r_f, v_theta_f, v_z_f)
    np.savetxt(str(time_stamp)+".txt", Q_t)
    print("Q_t: ",Q_t)
    print("Q Shape: ",Q_t.shape)

    fig = plt.figure()
    fig.set_figwidth(15)
    fig.set_figheight(8)

    plt.plot(Q_t,color='green', marker='o', linestyle='dashed',linewidth=1, markersize=5)
    plt.xlabel('Wavenumebr m at time t={}'.format(time_stamp))
    plt.ylabel('Eigenvalue')
    file_name = str(time_stamp)+".png"
    plt.savefig(file_name)
    #plt.show()

f.close()
end_time = time.time()
print('Time Duration: {}'.format(end_time - start_time))