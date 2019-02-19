import numpy as np
import emcee
import pyccl as ccl

import fittingtools as ft



def lnprior(theta):
    """Priors."""
    params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc", "b_hydro"]
    kwargs = dict(zip(params, theta))

    prior_test = (9 <= kwargs["Mmin"] <= 15)*\
                 (10 <= kwargs["M0"] <= 16)*\
                 (10 <= kwargs["M1"] <= 16)*\
                 (0.1 <= kwargs["sigma_lnM"] <= 1.0)*\
                 (0.5 <= kwargs["alpha"] <= 1.5)*\
                 (0.1 <= kwargs["fc"] <= 1.0)*\
                 (0.1 <= kwargs["b_hydro"] <= 0.9)*\
                 (kwargs["M0"] > kwargs["Mmin"])

    return 0.0 if prior_test else -np.inf



## INPUT ##
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)
data = ["2mpz, 2mpz", "2mpz, y_milca"]
l, cl, I, prof = ft.dataman(data, z_bin=1, cosmo=cosmo)



## MODEL ##
setup = {"cosmo"     : cosmo,
         "profiles"  : prof,
         "l_arr"     : l,
         "cl_arr"    : cl,
         "inv_covar" : I,
         "zrange"    : (0.001, 0.3)}


popt = [11.99, 14.94, 13.18, 0.26, 1.43, 0.54, 0.45]
ndim, nwalkers = len(popt), 100
pos = [popt + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, ft.lnprob,
                                args=(lnprior,), kwargs=setup)
ft.Neval = 1  # counter
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
        ax[i].plot(sampler.chain[j, :, i], "k-", lw=0.5, alpha=0.2)
    ax[i].get_yaxis().get_major_formatter().set_useOffset(False)
    ax[i].set_ylabel(yax[i], fontsize=15)
#fig.savefig("../images/MCMC_HOD_burn-in.pdf", dpi=600, bbox_inches="tight")

# Figure 2 (corner plot) #
cutoff = 50  # burn-in after cutoff steps
samples = sampler.chain[:, cutoff:, :].reshape((-1, ndim))

fig = corner.corner(samples, labels=yax, label_kwargs={"fontsize":15},
                    show_titles=True, quantiles=[0.16, 0.50, 0.84])
#fig.savefig("../images/MCMC_HOD_corner.pdf", dpi=600, bbox_inches="tight")

val = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
               zip(*np.percentile(samples, [16, 50, 84], axis=0))))
