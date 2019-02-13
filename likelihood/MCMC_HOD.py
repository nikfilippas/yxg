import numpy as np
import emcee
import pyccl as ccl

import profile2D
import pspec
import cosmotools as ct



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
    global cosmo, prof, l, cl, I, Neval

    lp = lnprior(theta)
    # Piecewise probability handling
    if not np.isfinite(lp):
        lnprob = -np.inf
    else:
        Cl = pspec.ang_power_spectrum(cosmo, l, prof, prof,
                                      zrange=(0.001, 0.3), **kwargs)

        # treat zero division (unphysical)
        if Cl is None:
            lnprob = -np.inf
        else:
            lnprob = lp + (-0.5*np.dot(cl-Cl, np.dot(I, cl-Cl)))

    print(Neval, theta); Neval += 1  # output trial parameter values
    return lnprob



## INPUT ##
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)

datasets = ["2mpz_2mpz", "2mpz_y_milca"]
covars = ["2mpz_2mpz_2mpz_2mpz", "2mpz_2mpz_2mpz_y_milca",
          "2mpz_y_milca_2mpz_2mpz", "2mpz_y_milca_2mpz_y_milca"]
dndz = "2MPZ_bin1"

l, cl, I = ct.dataman(cosmo, datasets, covars, dndz)



## MODEL ##
nz = "../analysis/data/dndz/2MPZ_bin1.txt"
prof = profile2D.HOD(nz_file=nz)

popt = [11.99, 14.94, 13.18, 0.32, 1.41, 0.56]
ndim, nwalkers = len(popt), 100
pos = [popt + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
sampler.run_mcmc(pos, 500)



### PLOTS ##
#import matplotlib.pyplot as plt
#import corner
#
#yax = ["$\\mathrm{M_{min}}$", "$\\mathrm{M_0}$", "$\\mathrm{M_1}$",
#       "$\\mathrm{\\sigma_{\\ln M}}$", "$\\mathrm{\\alpha}$", "$\\mathrm{fc}$"]
#
## Figure 1 (burn-in histogram) #
#fig, ax = plt.subplots(6, 1, sharex=True, figsize=(5, 10))
#ax[-1].set_xlabel("step number", fontsize=15)
#
#
#for i in range(ndim):
#    for j in range(nwalkers):
#        ax[i].plot(sampler.chain[j, :, i], "k-", lw=0.5, alpha=0.2)
#    ax[i].get_yaxis().get_major_formatter().set_useOffset(False)
#    ax[i].set_ylabel(yax[i], fontsize=15)
#fig.savefig("../images/MCMC_HOD_burn-in.pdf", dpi=600, bbox_inches="tight")
#
## Figure 2 (corner plot) #
#cutoff = 50  # burn-in after cutoff steps
#samples = sampler.chain[:, cutoff:, :].reshape((-1, ndim))
#
#fig = corner.corner(samples, labels=yax, label_kwargs={"fontsize":15},
#                    show_titles=True, quantiles=[0.16, 0.50, 0.84])
#fig.savefig("../images/MCMC_HOD_corner.pdf", dpi=600, bbox_inches="tight")
#
#val = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#               zip(*np.percentile(samples, [16, 50, 84], axis=0))))
