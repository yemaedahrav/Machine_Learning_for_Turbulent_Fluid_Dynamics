import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

a = np.loadtxt("s3_Q0.txt", dtype=np.float32)

a = np.asarray(a)
print(a)
print(a.shape)

b = np.resize(a, (1,128))
print(b)
print(b.shape)
b = b.ravel()
print(b.shape)

b[0] = 0

f = plt.figure()
f.set_figwidth(20)
f.set_figheight(7)

plt.plot(b,color='teal', marker='*', linestyle='dotted',linewidth=0.8, markersize=3)
plt.xlabel('Wavenumebr m at time t={}'.format(0))
plt.ylabel('Eigenvalue')
# plt.xticks(np.arange())
file_name = "s3_0_adjusted"+".png"
plt.savefig(file_name, dpi=500)
plt.show()
