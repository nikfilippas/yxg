"""
temp file for calcs and unstable test code
"""


import numpy as np
import pyccl as ccl

from cosmotools import R_Delta
import profile2D


# Cosmology Definition
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)

a = 1
Delta = 500


rmin, rmax = 1e-4, 5  # [R_Delta]
kmin, kmax = 2*np.pi/rmax, 2*np.pi/rmin


## minimum and maximum halo mass
#Mmin, Mmax = 1e6, 1e16
#Rmin, Rmax = 1e-4, 5
## wish to probe distances between 1e-4 and 5 R_Delta for *any* halo mass
## since R_Delta is an increasing function of M
#rmin = Rmin*R_Delta(cosmo, Mmin, Delta)
#rmax = Rmax*R_Delta(cosmo, Mmax, Delta)
## angular wavenumber is 2pi/lambda
#kmin = 2*np.pi/rmax
#kmax = 2*np.pi/rmin

# so, values of k probed, logarithmically span [kmin, kmax]
k_arr = np.logspace(np.log10(kmin), np.log10(kmax), 100)

p1 = profile2D.Arnaud()
p2 = profile2D.NFW()
