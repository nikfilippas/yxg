import numpy as np
import emcee
import pyccl as ccl

import profile2D
import pspec



def lnprior(theta):
    """Priors."""
    params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc"]
    kwargs = dict(zip(params, theta))

    prior_test = (6 <= kwargs["Mmin"] <= 17)*\
                 (6 <= kwargs["M0"] <= 17)*\
                 (6 <= kwargs["M1"] <= 17)*\
                 (0.1 <= kwargs["sigma_lnM"] <= 1.0)*\
                 (0.5 <= kwargs["alpha"] <= 1.5)*\
                 (0.1 <= kwargs["fc"] <= 1.0)

    return 0.0 if prior_test else -np.inf



Neval = 1
def lnprob(theta):
    """Probability distribution to be sampled."""
    params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc"]
    kwargs = dict(zip(params, theta))
    global cosmo, prof, l, cl, Neval

    lp = lnprior(theta)
    # Piecewise probability handling
    if not np.isfinite(lp):
        lnprob = -np.inf
    else:
        Cl = pspec.ang_power_spectrum(cosmo, l, prof, prof,
                                      zrange=(0.001, 0.3), zpoints=64, **kwargs)

        # treat zero division (unphysical)
        if Cl is None:
            lnprob = -np.inf
        else:
            lnprob = lp + (-0.5*np.dot(cl-Cl, np.dot(I, cl-Cl)))

    print(Neval, theta); Neval += 1  # output trial parameter values
    return lnprob



## DATA ##
# science
data = np.load("../analysis/out_ns512_linlog/cl_2mpz_y_milca.npz")  # clyg
# covariances
cov = np.load("../analysis/out_ns512_linlog/cov_2mpz_y_milca_2mpz_y_milca.npz")  # clyg

# x-data
l = data["leff"]
mask = l < 260
l = l[mask]
# y-data
cl = data["cell"] - data["nell"]
cl = cl[mask]
# error bars
covar = cov["cov"]
covar = covar[mask, :][:, mask]
err = np.sqrt(np.diag(covar))
I = np.linalg.inv(covar)  # inverse covariance



## MODEL ##
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)

nz = "../analysis/data/dndz/2MPZ_bin1.txt"
prof = profile2D.HOD(nz_file=nz)

ndim, nwalkers = 6, 100
popt = [12.00287818, 14.94087941, 13.18144554, 0.27649579, 1.43902899, 0.57055288]
pos = [popt + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
sampler.run_mcmc(pos, 500)



## PLOTS ##
import matplotlib.pyplot as plt
import corner

yax = ["$\\mathrm{M_{min}}$", "$\\mathrm{M_0}$", "$\\mathrm{M_1}$",
       "$\\mathrm{\\sigma_{\\ln M}}$", "$\\mathrm{\\alpha}$", "$\\mathrm{fc}$"]

# Figure 1 (burn-in histogram) #
fig, ax = plt.subplots(6, 1, sharex=True, figsize=(5, 10))
ax[-1].set_xlabel("step number", fontsize=15)


for i in range(ndim):
    for j in range(nwalkers):
        ax[i].plot(sampler.chain[j, :, i], "k-", lw=0.5)
    ax[i].get_yaxis().get_major_formatter().set_useOffset(False)
    ax[i].set_ylabel(yax[i], fontsize=15)


# Figure 2 (corner plot) #
cutoff = 0  # burn-in after cutoff steps
samples = sampler.chain[:, cutoff:, :].reshape((-1, ndim))

fig = corner.corner(samples, labels=yax)

val = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
               zip(*np.percentile(samples, [16, 50, 84], axis=0))))
