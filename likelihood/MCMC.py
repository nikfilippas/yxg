"""
- add progressbar (tqdm) via callback function in Parallel
"""
import numpy as np
import pyccl as ccl

import multiprocessing as mp
from tqdm import tqdm

import fittingtools as ft



# survey properties: name, zrange, bin
sprops = {"2mpz"     :  [(0.0005, 0.0745), 1],
          "wisc_b1"  :  [(0.0005, 0.1405), 1],
          "wisc_b2"  :  [(0.0215, 0.1575), 2],
          "wisc_b3"  :  [(0.0505, 0.1785), 3],
          "wisc_b4"  :  [(0.0735, 0.2035), 4],
          "wisc_b5"  :  [(0.0895, 0.2355), 5]}
cosmo = ccl.Cosmology(Omega_c=0.26066676,
                      Omega_b=0.048974682,
                      h=0.6766,
                      sigma8=0.8102,
                      n_s=0.9665)

popt = [11.99, 14.94, 13.18, 0.26, 1.43, 0.54, 0.45]
nwalkers, nsteps = 100, 500

def sampler(sur):
    return ft.MCMC(sur, sprops, cosmo, popt,
                   ft.lnprob, args=(ft.lnprior,), nwalkers=nwalkers, nsteps=nsteps)


with mp.Pool(mp.cpu_count()) as p:
    results = p.map(sampler, list(sprops.keys()))



## RESULTS & OUTPUT ##
cutoff = 100  # burn-in after cutoff steps
for s, sur in enumerate(sprops.keys()):
    samples = results[s].chain[:, cutoff:, :].reshape((-1, len(popt)))

    val = list(map(lambda v: (v[1], v[1]-v[0], v[2]-v[1]),
                   zip(*np.percentile(samples, [16, 50, 84], axis=0))))
    np.save("fit_vals/"+sur, np.array(val).T)



## PLOTS ##
import matplotlib.pyplot as plt
import corner

yax = ["$\\mathrm{M_{min}}$", "$\\mathrm{M_0}$", "$\\mathrm{M_1}$",
       "$\\mathrm{\\sigma_{\\ln M}}$", "$\\mathrm{\\alpha}$", "$\\mathrm{fc}$",
       "$\\mathrm{b_{hydro}}$"]

# Figure 1 (burn-in histogram) #
for s, sur in sprops.keys():
    fig, ax = plt.subplots(len(popt), 1, sharex=True, figsize=(7, 12))
    ax[-1].set_xlabel("step number", fontsize=15)

    for i in range(results[s].dim):
        for j in range(results[s].k):
            ax[i].plot(results[s].chain[j, :, i], "k-", lw=0.5, alpha=0.2)
        ax[i].get_yaxis().get_major_formatter().set_useOffset(False)
        ax[i].set_ylabel(yax[i], fontsize=15)
    plt.tight_layout()
    fig.savefig("../images/MCMC/MCMC_steps_%s.pdf" % sur,
                dpi=600, bbox_inches="tight")

# Figure 2 (corner plot) #
for s, sur in enumerate(sprops.keys()):
    samples = results[s].chain[:, cutoff:, :].reshape((-1, len(popt)))
    fig = corner.corner(samples, labels=yax, quantiles=[0.16, 0.50, 0.84],
                        show_titles=True, label_kwargs={"fontsize":15})
    fig.savefig("../images/MCMC/MCMC_%s.pdf" % sur,
                dpi=600, bbox_inches="tight")