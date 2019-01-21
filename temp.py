"""
temp file for calcs and unstable test code
"""


import numpy as np
import pyccl as ccl

import profile2D


# Cosmology Definition
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.67, sigma8=0.8, n_s=0.96)

a = 1
Delta = 500

rmin, rmax = 1e-4, 5  # [R_Delta]
kmin, kmax = 2*np.pi/rmax, 2*np.pi/rmin
# so, values of k probed, logarithmically span [kmin, kmax]
k_arr = np.logspace(np.log10(kmin), np.log10(kmax), 100)
M_arr = np.logspace(6, 17, 256)  # masses sampled

p1 = profile2D.Arnaud()
p2 = profile2D.Arnaud()
p3 = profile2D.HOD()
