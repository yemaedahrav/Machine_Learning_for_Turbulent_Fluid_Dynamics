import sys
import time
import numpy as np  
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py

f = h5py.File('state_variables24_10_g_s3.h5', 'r')

def printname(name):
    print(name)
    time.sleep(2)
f.visit(printname)


r = f['scales']['r']['1.0']
z = f['scales']['z']['1.0']
th = f['scales']['th']['1.0']
t = f['scales']['sim_time']

th = np.asarray(th)
z = np.asarray(z)
r = np.asarray(r)
t = np.asarray(t)

ul = f['tasks']['ul'][0]
time.sleep(2)

uh = f['tasks']['uh'][0]
time.sleep(2)

vl = f['tasks']['vl'][0]
time.sleep(2)

vh = f['tasks']['vh'][0]
time.sleep(2)

wl = f['tasks']['wl'][0]
time.sleep(2)

wh = f['tasks']['wh'][0]
time.sleep(2)

ul = np.asarray(ul)
time.sleep(2)

uh = np.asarray(uh)
time.sleep(2)

vl = np.asarray(vl)
time.sleep(2)

vh = np.asarray(vh)
time.sleep(2)

wl = np.asarray(wl)
time.sleep(2)

wh = np.asarray(wh)
time.sleep(2)

N_r = r.shape[0]
N_th = th.shape[0]
N_z = z.shape[0]
N_t = t.shape[0]

print("N_r: ", N_r)
print("N_th: ", N_th)
print("N_z: ", N_z)
print("N_t: ", N_t)

v_r = ul + uh
v_theta = vl + vh
v_z = wl + wh
print("v_r shape: ",v_r.shape)

## The dimensions correspond to a is t, r, theta, z ##


## This 2-Dimensional array stores the eigenvalues for each corresponding wavenumber m and time t  pair ##
Q = np.zeros((N_th, 1))

print("Q Shape: ", Q.shape)
print("Q Value", Q)

# for time_stamp in range(N_t):

v_r_f = np.fft.fft(v_r,axis=1)
time.sleep(2)

v_theta_f = np.fft.fft(v_theta,axis=1)
time.sleep(2)

v_z_f = np.fft.fft(v_z,axis=1)
time.sleep(2)

for m in range(N_th):
    print("Wavenumber m: ",m)
    eigen_val = 0
    time.sleep(0.005)
    for R in range(1,N_r):
        print("R: ",R)
        for Z in range(1,N_z):
            print("Z: ",Z)
            Q_v_r = v_r_f[Z][m][R]
            Q_v_theta = v_theta_f[Z][m][R]
            Q_v_z = v_z_f[Z][m][R]
            power_amplitude = (Q_v_r)*np.conj(Q_v_r) + (Q_v_theta)*np.conj(Q_v_theta) + (Q_v_z)*np.conj(Q_v_z)
            ## print(power_amplitude)   just a check to see that all these values are real ##
            time.sleep(0.005)
            eigen_val += r[R]*(r[R]-r[R-1])*(z[Z]-z[Z-1])*(power_amplitude)
    Q[m][0] = 2*(np.pi)*eigen_val 


# for time_slice in range(N_t):
Q_t = np.asarray(Q[:,0])
print("Timeslice: ", 0 , " ",Q_t.shape)
print("Value of Q_t: ", Q_t)
plt.plot(Q_t,color='green', marker='o', linestyle='dashed',linewidth=1, markersize=5)
plt.xlabel('Wavenumebr m at time t={}'.format(0))
plt.ylabel('Eigenvalue')
file_name = str(t[0])+".png"
## plt.show()
plt.savefig(file_name)