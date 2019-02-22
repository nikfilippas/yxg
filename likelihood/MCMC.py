"""
``zrange`` for different redshift bins:

=======================================================
 SURVEY       ZRANGE      ||  SURVEY       ZRANGE     |
2mpz:     (0.001, 0.300)  || wisc_b3:  (0.020, 0.500) |
wisc_b1:  (0.001, 0.320)  || wisc_b4:  (0.050, 0.600) |
wisc_b2:  (0.005, 0.370)  || wisc_b5:  (0.070, 0.700) |
=======================================================
"""
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



# INPUT #
survey = "2mpz"

# survey properties: name, zrange, bin
sprops = {"2mpz"   :  [(0.001, 0.300), 1],
          "wisc_b1":  [(0.001, 0.320), 1],
          "wisc_b2":  [(0.005, 0.370), 2],
          "wisc_b3":  [(0.020, 0.500), 3],
          "wisc_b4":  [(0.050, 0.600), 4],
          "wisc_b5":  [(0.070, 0.700), 5]}



## DATA MANIPULATION ##
cosmo = ccl.Cosmology(Omega_c=0.26066676, Omega_b=0.048974682, h=0.6766,
                      sigma8=0.8102, n_s=0.9665)
data = [survey+","+survey, survey+","+"y_milca"]
l, cl, _, I, prof = ft.dataman(data, z_bin=sprops[survey][1], cosmo=cosmo)

setup = {"cosmo"     : cosmo,
         "profiles"  : prof,
         "l_arr"     : l,
         "cl_arr"    : cl,
         "inv_covar" : I,
         "zrange"    : sprops[survey][0]}

popt = [11.99, 14.94, 13.18, 0.26, 1.43, 0.54, 0.45]



## MODEL ##
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
       "$\\mathrm{\\sigma_{\\ln M}}$", "$\\mathrm{\\alpha}$", "$\\mathrm{fc}$",
       "$\\mathrm{b_{hydro}}$"]

# Figure 1 (burn-in histogram) #
fig, ax = plt.subplots(len(popt), 1, sharex=True, figsize=(7, 12))
ax[-1].set_xlabel("step number", fontsize=15)


for i in range(ndim):
    for j in range(nwalkers):
        ax[i].plot(sampler.chain[j, :, i], "k-", lw=0.5, alpha=0.2)
    ax[i].get_yaxis().get_major_formatter().set_useOffset(False)
    ax[i].set_ylabel(yax[i], fontsize=15)
plt.tight_layout()
fig.savefig("../images/MCMC_steps_%s.pdf" % survey, dpi=600, bbox_inches="tight")

# Figure 2 (corner plot) #
cutoff = 100  # burn-in after cutoff steps
samples = sampler.chain[:, cutoff:, :].reshape((-1, ndim))

fig = corner.corner(samples, labels=yax, label_kwargs={"fontsize":15},
                    show_titles=True, quantiles=[0.16, 0.50, 0.84])
fig.savefig("../images/MCMC_%s.pdf" % survey, dpi=600, bbox_inches="tight")

val = list(map(lambda v: (v[1], v[1]-v[0], v[2]-v[1]),
               zip(*np.percentile(samples, [16, 50, 84], axis=0))))
np.save("fit_vals/"+survey, np.array(val).T)