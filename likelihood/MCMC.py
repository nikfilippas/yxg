import numpy as np
import emcee
import pyccl as ccl

import profile2D
import pspec



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
    if not prior_test:  # deal with priors
        lnprob = -np.inf
    else:
        Cl = pspec.ang_power_spectrum(cosmo, ells, prof, prof, **kwargs)
        if Cl is None:  # deal with zero division (unphysical)
            lnprob = -np.inf
        else :
            lnprob = -0.5*np.dot(cells-Cl, np.dot(I, cells-Cl))

#    print(args)  # output trial parameter values
    return -lnprob


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
I = np.linalg.inv(covar)  # inverse covariance


## MODEL ##
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)

nz = "../analysis/data/dndz/2MPZ_bin1.txt"
prof = profile2D.HOD(nz_file=nz)

ndim, nwalkers = 6, 100
ivar = 1 / np.random.rand(ndim)
p0 = [12.061706, 12.881403, 13.567549, 0.632100, 1.238674, 0.644111]

sampler = emcee.EnsembleSampler(nwalkers, ndim, func, args=[ivar])
sampler.run_mcmc(p0, 1000)