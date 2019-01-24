import numpy as np
import matplotlib.pyplot as plt
import emcee
import pyccl as ccl

import pspec
import profile2D



cell = np.load("cell.npy")
cov = np.load("cov.npy")
leff = np.load("leff.npy")

## MODEL ##
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)

nz = "data/2MPZ_histog_Lorentz_2.txt"
prof = profile2D.HOD(nz_file=nz)

l_arr = np.arange(260)
kwargs = {"Mmin"      : 1e12,
          "M0"        : 10**12.2,
          "M1"        : 10**13.65,
          "sigma_lnM" : 0.5,
          "alpha"     : 1.0,
          "fc"        : 0.8}
Cl = pspec.ang_power_spectrum(cosmo, l_arr, prof, prof, is_zlog=False, **kwargs)

plt.loglog(l_arr, Cl)

