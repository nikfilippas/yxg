"""
BENCHMARK

-   Varying the number of z integration points from 128 to 32 has negligible
    shift in the highest multiples l~100, of the order of 0.02%.
-   Varying the number of m integration points from 256 to 128, 64 and 32 results
    to shift in the highest multplies ~100, of the order of 0.16%, 2.89%,
    and 8.38% respectively.
-   Varying both z and m integration points adds an additional 0.02% offset in
    the highest multiples, to the already existing m-offset.

**  Listed are the tested combinations of [zpoints, mpoints], (times faster),
    for how many times the algorithm is sped up:

       [128., 256.],    (1.)
       [128., 128.],    (2.01934416)
       [128.,  64.],    (3.65305686)
       [128.,  32.],    (5.59618493)
       [ 64., 256.],    (2.01872085)
       [ 64., 128.],    (4.05803101)
       [ 64.,  64.],    (7.22972731)
       [ 64.,  32.],    (11.12368002)
       [ 32., 256.],    (3.81940437)
       [ 32., 128.],    (7.72527809)
       [ 32.,  64.],    (14.25514526)
       [ 32.,  32.],    (22.15638888).

>>  It is thus chosen that the algorithm runs with 32 z-points and 128 m-points
    for a total ~8-fold speed increase.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import time
import pyccl as ccl

import profile2D
import pspec



## MODEL ##
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)
nz = "../../analysis/data/dndz/2MPZ_bin1.txt"
prof = profile2D.HOD(nz_file=nz)
l_arr = np.arange(260)
kwargs = {"Mmin"      : 12.00287818,
          "M0"        : 14.94087941,
          "M1"        : 13.18144554,
          "sigma_lnM" : 0.27649579,
          "alpha"     : 1.43902899,
          "fc"        : 0.57055288}



## VARY Z ##
fig1, ax1 = plt.subplots(1, 1)
ax1.set_xlabel("$\\ell$", fontsize=15)
ax1.set_ylabel("$C^{gg}_{\\ell}$", fontsize=15)

num1 = 3
z_arr = np.geomspace(128, 32, num1)
col1 = [viridis(i) for i in np.linspace(0, 0.9, len(z_arr))]

times1 = np.empty(num1)
for i, z in enumerate(z_arr):

    t0 = time.time()
    for j in range(10):
            Cl = pspec.ang_power_spectrum(cosmo, l_arr, (prof, prof),
                               zrange=(0.001, 0.3), zpoints=z, is_zlog=True,
                               logMrange=(6, 17), mpoints=256,
                               include_1h=True, include_2h=True, **kwargs)
    t1 = time.time() - t0
    times1[i] = t1/10

    ax1.loglog(l_arr, Cl, c=col1[i])
ratio1 = 1/(times1/times1[0])



## VARY M ##
fig2, ax2 = plt.subplots(1, 1)
ax2.set_xlabel("$\\ell$", fontsize=15)
ax2.set_ylabel("$C^{gg}_{\\ell}$", fontsize=15)

num2 = 4
m_arr = np.geomspace(256, 32, num2)
col2 = [viridis(i) for i in np.linspace(0, 0.9, len(m_arr))]

times2 = np.empty(num2)
for i, m in enumerate(m_arr):

    t0 = time.time()
    for j in range(10):
            Cl = pspec.ang_power_spectrum(cosmo, l_arr, (prof, prof),
                               zrange=(0.001, 0.3), zpoints=128, is_zlog=True,
                               logMrange=(6, 17), mpoints=m,
                               include_1h=True, include_2h=True, **kwargs)
    t1 = time.time() - t0
    times2[i] = t1/10

    ax2.loglog(l_arr, Cl, c=col2[i])
ratio2 = 1/(times2/times2[0])



## VARY Z, M ##
fig3, ax3 = plt.subplots(1, 1)
ax3.set_xlabel("$\\ell$", fontsize=15)
ax3.set_ylabel("$C^{gg}_{\\ell}$", fontsize=15)

t_arr = np.array(np.meshgrid(z_arr, m_arr)).T.reshape(-1, 2)
col3 = [viridis(i) for i in np.linspace(0, 0.9, len(t_arr))]

times3 = np.empty(num1*num2)
for i, t in enumerate(t_arr):

    t0 = time.time()
    for j in range(10):
            Cl = pspec.ang_power_spectrum(cosmo, l_arr, (prof, prof),
                               zrange=(0.001, 0.3), zpoints=t[0], is_zlog=True,
                               logMrange=(6, 17), mpoints=t[1],
                               include_1h=True, include_2h=True, **kwargs)
    t1 = time.time() - t0
    times3[i] = t1/10

    ax3.loglog(l_arr, Cl, c=col3[i])
ratio3= 1/(times3/times3[0])



print(ratio1)
print(ratio2)
print(ratio3)