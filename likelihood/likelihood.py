import numpy as np
from scipy.optimize import minimize
import pyccl as ccl

import pspec
import profile2D


def func(args):
    """The function to be minimised."""
    params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc"]
    kwargs = dict(zip(params, args))

    prior_test = (6 <= kwargs["Mmin"] <= 17)*\
                 (6 <= kwargs["M0"] <= 17)*\
                 (6 <= kwargs["M1"] <= 17)*\
                 (0.1 <= kwargs["sigma_lnM"] <= 1.0)*\
                 (0.5 <= kwargs["alpha"] <= 1.5)*\
                 (0.1 <= kwargs["fc"] <= 1.0)

    # Piecewise probability handling
    if not prior_test:
        lnprob = -np.inf
    else:
        Cl = pspec.ang_power_spectrum(cosmo, ells, prof, prof,
                                      zrange=(0.001,0.3), zpoints=64,
                                      is_zlog=False, **kwargs)
        if Cl is None:
            lnprob = -np.inf
        else :
            lnprob = -0.5*np.dot(cells-Cl, np.dot(I, cells-Cl))

    print(args,-2*lnprob,len(ells))  # output trial parameter values
    return -lnprob


Neval = 1  # display number of evaluations
def callbackf(X):
    global Neval
    print("{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   \
          {4: 3.6f}   {5: 3.6f}   {6: 3.6f}".format(Neval, X[0], X[1], X[2],
                                                           X[3], X[4], X[5]))
    Neval += 1



## DATA ##
dcl = np.load("../analysis/out_ns512_linlog/cl_2mpz_2mpz.npz")
dcov = np.load("../analysis/out_ns512_linlog/cov_2mpz_2mpz_2mpz_2mpz.npz")

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

nz = "../analysis/data/dndz/2MPZ_bin1.txt"
prof = profile2D.HOD(nz_file=nz)

I = np.linalg.inv(covar)  # inverse covariance


p0 = [12.3, 12.5, 13.65, 0.5, 1.1, 0.8]
res = minimize(func, p0, method="Powell", callback=callbackf)

## PLOTS ##
import matplotlib.pyplot as plt

def dataplot(xdata, ydata, yerr, popt):
    params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc"]
    kwargs = dict(zip(params, popt))
    Cl = pspec.ang_power_spectrum(cosmo, ells, prof, prof,
                                  zrange=(0.001,0.3), zpoints=64,
                                  is_zlog=False,**kwargs)

    plt.figure()
    plt.xlabel('$\\ell$',fontsize=15)
    plt.ylabel('$C_\\ell$',fontsize=15)

    plt.errorbar(ells, cells, err_ell, fmt="rs")
    plt.loglog(ells, Cl, "k-", lw=3)
    plt.show()


dataplot(ells, cells, err_ell, res.x)

#kwargs={"Mmin"      : 12.3,
#        "M0"        : 12.5,
#        "M1"        : 13.65,
#        "sigma_lnM" : 0.5,
#        "alpha"     : 1.1,
#        "fc"        : 0.8}
