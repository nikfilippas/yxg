import numpy as np
from scipy.optimize import minimize
from scipy.optimize import Bounds
import emcee
import pyccl as ccl

import pspec
import profile2D



## DATA ##
dcl = np.load("cl_2mpz_2mpz.npz")
dcov = np.load("cov_2mpz_2mpz_2mpz_2mpz.npz")

# x-data
ells = dcl['leff']
mask = ells < 260  # mask
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




## MODEL ##
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)

nz = "data/2MPZ_histog_Lorentz_2.txt"
prof = profile2D.HOD(nz_file=nz)

kwargs = {"Mmin"      : 12,
          "M0"        : 12.2,
          "M1"        : 13.65,
          "sigma_lnM" : 0.5,
          "alpha"     : 1.0,
          "fc"        : 0.8}
Cl = pspec.ang_power_spectrum(cosmo, ells, prof, prof, is_zlog=False, **kwargs)



I = np.linalg.inv(covar)  # inverse covariance

def func(args):
    """The function to be minimised."""
    params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc"]
    kwargs = dict(zip(params, args))

    Cl = pspec.ang_power_spectrum(cosmo, ells, prof, prof, is_zlog=False, **kwargs)
    chi2 = np.dot(cells-Cl, np.dot(I, cells-Cl))
    lnprob = -0.5 * chi2
    return -lnprob

p0 = [12, 12.2, 13.65, 0.5, 1.0, 0.8]
bds = [(6, 17),     # Mmin
       (6, 17),     # M0
       (6, 17),     # M1
       (0.1, 1),    # sigma_lnM
       (0.5, 1.5),  # alpha
       (0.1, 1.0)]  # fc
res = minimize(func, p0, method="Powell")


### PLOTS ##
#import matplotlib.pyplot as plt
#from matplotlib.colors import SymLogNorm
#from pylab import cm
## PLOT 1
#plt.figure()
#plt.errorbar(ells, cells, yerr=err_ell, fmt='rs', ms=5)
#plt.loglog()
#plt.xlabel('$\\ell$',fontsize=15)
#plt.ylabel('$C_\\ell$',fontsize=15)
## PLOT 2
#plt.loglog(ells, Cl)
## PLOT 3
#P = plt.subplots(1, 2)
#[x.imshow(X.T, cmap=cm.viridis, norm=SymLogNorm(linthresh=1e-17, vmin=X.min(), vmax=X.max())) for x, X in zip(P[1], [covar, I])]
