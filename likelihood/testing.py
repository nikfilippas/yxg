"""test code"""
import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt

import profile2D
import pspec



def dataplot(xdata, ydata, yerr, popt):
    params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc"]
    kwargs = dict(zip(params, popt))
    Cl = pspec.ang_power_spectrum(cosmo, ells, prof, prof, **kwargs)

    plt.figure()
    plt.xlabel('$\\ell$',fontsize=15)
    plt.ylabel('$C_\\ell$',fontsize=15)

    plt.errorbar(ells, cells, err_ell, fmt="rs")
    plt.loglog(ells, Cl, lw=3)


## DATA ##
dcl = np.load("../analysis/out_ns512_linlog/cl_2mpz_2mpz.npz")
dcov = np.load("../analysis/out_ns512_linlog/cov_2mpz_2mpz_2mpz_2mpz.npz")

# x-data
ells = dcl['leff']
mask = ells < 260
ells = ells[mask]
# y-data
cells_with_noise = dcl['cell']
nells = dcl['nell']
cells = cells_with_noise-nells
cells = cells[mask]
# error bars
covar=dcov['cov']
covar = covar[mask, :][:, mask]
err_ell = np.sqrt(np.diag(covar))

## PROFILE ##
nz = "../analysis/data/dndz/2MPZ_bin1.txt"
prof = profile2D.HOD(nz_file=nz)
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)

p0 = [12.061706, 12.881403, 13.567549, 0.632100, 1.238674, 0.644111]
