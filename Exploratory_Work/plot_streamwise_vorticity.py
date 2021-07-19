# Load python packages and Dedalus
import numpy as np  
import time
import matplotlib.pyplot as plt
import h5py

import matplotlib.colors as colors
from numba import jit,njit, literal_unroll
import sys


start_time = time.time()

# Run it for the first time to check the output path. Once it's done this can be commented out.


# Number of time indexing
nu = 0.00020704166
dpdz = -0.0098763844 
r_in = 0.5
r_out = 1

class MidpointNormalize(colors.Normalize):
	"""
	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
#This maps minimum, midpoint and maximum values to 0, 0.5 and 1 respectively i.e.


@njit(fastmath = True)
def vor(r,th,z,v):
    vor = np.zeros(v.shape)
    for i, r_ in enumerate (r):
        for j, th_ in enumerate (th):
            for k, z_ in enumerate (z):
                vor[k,j,i] = v[k,j,i]/r[i]
    return vor


j = int(float(sys.argv[1])) # looping j to determine which file to read/plot

"""
filelist = ["/users/czhang54/scratch/ret78_g/ret78_g_s1.h5", "/users/czhang54/scratch/ret78_2_g/ret78_2_g_s1.h5",
        "/users/czhang54/scratch/ret78_3_g/ret78_3_g_s1.h5"]
"""

filelist = ["/home/amey/Documents/ML4SCI/Week1/smallQL_c_s1_p0.h5", "/home/amey/Documents/ML4SCI/Week1/small_QL_g_s1_p0.h5"]
filelist = np.asarray(filelist)
namelist = ["1_s1_{}.png","2_s1_{}.png"]
name = namelist[j]




# Plot
with h5py.File(filelist[j], mode='r') as file:
      
    t = file['scales']['sim_time']

    if j==1:
        r = file['scales']['r']['1.0']
        z = file['scales']['z']['1.0']
        th = file['scales']['th']['1.0']
    else:
        r = file['scales']['r']
        z = file['scales']['z']
        th = file['scales']['th']
  
    th = np.asarray(th)
    z = np.asarray(z)
    r = np.asarray(r)
    r1=r[:]
    z1=z[:]
    t1=t[:]

    v_m = 1.0372661594647228 # Avg z-direction velocity calculated beforhand to normalize the field here

    r_,z_=np.meshgrid(z1,r1)

    w_array = np.zeros(r_.shape)
    
    for index in range(0,len(t1),10):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if j==1:
            ul = file['tasks']['ul'][index,:,:,:]
            uh = file['tasks']['uh'][index,:,:,:]
            vl = file['tasks']['vl'][index,:,:,:]
            vh = file['tasks']['vh'][index,:,:,:]
            wl = file['tasks']['wl'][index,:,:,:]
            wh = file['tasks']['wh'][index,:,:,:]
        else:
            ul = file['tasks']['ulc'][index,:,:,:]
            uh = file['tasks']['uhc'][index,:,:,:]
            vl = file['tasks']['vlc'][index,:,:,:]
            vh = file['tasks']['vhc'][index,:,:,:]
            wl = file['tasks']['wlc'][index,:,:,:]
            wh = file['tasks']['whc'][index,:,:,:]


        vel = np.asarray([ul,uh,vl,vh,wl,wh])

        for u in vel:
            u = np.asarray(u)

        u = ul + uh
        v = vl + vh
        w = wl + wh

        u = u/v_m
        v = v/v_m
        w = w/v_m

        # Define streamwise vorticity xi
        dvdr = np.gradient(v,axis = 2)
        dudth = np.gradient(u, axis = 1)
        xi = dvdr + vor(r,th,z,v) + vor(r,th,z,dudth)
        xi_show = xi[:,0,:]


        v_min = xi.min()
        v_max = xi.max()
        mid_val = 0

        p=plt.pcolormesh(r_,z_,np.transpose(xi_show),cmap='RdBu_r',clim=(v_min, v_max), norm=MidpointNormalize(midpoint=mid_val,vmin=v_min, vmax=v_max))

        plt.xlabel('z')
        plt.ylabel('r')
        plt.title("Streamwise Vorticity at theta = 0 and t = {:3.1f}".format(t[index]*v_m/0.5))

        ax.set_aspect('12')
        plt.colorbar(orientation='horizontal')
        
        # plt.savefig("/users/czhang54/scratch/ret78_g/xi/xi_"+name.format(index))
        plt.savefig("/home/amey/Documents/ML4SCI/Week1/"+name.format(index))

        plt.close()

end_time = time.time()
print('Run time: {}'.format(end_time - start_time))