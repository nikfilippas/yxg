import os
os.chdir("../..")
import numpy as np
import pyccl as ccl
from analysis.params import ParamRun
from model.data import DataManager
from likelihood.like import Likelihood
from likelihood.sampler import Sampler
from model.theory import get_theory
from model.power_spectrum import HalomodCorrection
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


# Theory predictor wrapper
class thr(object):
    def __init__(self, d):
        self.d = d

    def th(self, pars):
        return get_theory(p, self.d, cosmo, hm_correction=hm_correction,
                          selection=sel, **pars)

    def th1h(self, pars):
        return get_theory(p, self.d, cosmo, hm_correction=hm_correction,
                          selection=sel, include_2h=False, include_1h=True,
                          **pars)

    def th2h(self, pars):
        return get_theory(p, self.d, cosmo, hm_correction=hm_correction,
                          selection=sel, include_2h=True, include_1h=False,
                          **pars)



fnames = ["params.yml", "params_test_dndz.yml"]
p, q = [ParamRun(fname) for fname in fnames]
cosmo = p.get_cosmo()

hm_correction = HalomodCorrection(cosmo)
sel = None

surveys = ["2mpz"] + ["wisc%d" % i for i in range(1, 6)]
sci = [r"$\mathrm{2MPZ}$"] + \
      [r"$\mathrm{WI \times SC}$ - $\mathrm{%d}$" % i for i in range(1, 6)]


f = plt.figure(figsize=(8, 12))
gs_main = GridSpec(6, 2, figure=f)


for s, v in enumerate(p.get("data_vectors")):

    # Construct data vector and covariance
    d = DataManager(p, v, cosmo)
    g = DataManager(q, v, cosmo)

    thd = thr(d)
    thg = thr(g)

    z, nz = np.loadtxt(d.tracers[0][0].dndz, unpack=True)
    zmean = np.average(z, weights=nz)
    chi = ccl.comoving_radial_distance(cosmo, 1/(1+zmean))
    kmax = p.get("mcmc")["kmax"]
    lmax = kmax*chi - 0.5

    # Set up likelihood
    lik = Likelihood(p.get('params'), d.data_vector, d.covar, thd.th)
    # Set up sampler
    sam = Sampler(lik.lnprob, lik.p0, lik.p_free_names,
                  p.get_sampler_prefix(v['name']), p.get('mcmc'))

    # Read chains and best-fit
    sam.get_chain()
    sam.update_p0(sam.chain[np.argmax(sam.probs)])

    params = lik.build_kwargs(sam.p0)

    # Array of multipoles
    ls = np.array(d.ells)

    # Indices used in the analysis
    def unequal_enumerate(a):
        """Returns indices of all elements in nested arrays."""
        indices = []
        ind0 = 0
        for l in a:
            sub = [x for x in range(ind0, ind0+len(l))]
            indices.append(sub)
            ind0 += len(l)
        return np.array(indices)

    def unwrap(arr, indices):
        arr_out = []
        for i in indices:
            arr_out.append(arr[i])
        return arr_out

    def eval_and_unwrap(pars, func, indices):
        return unwrap(func(pars), indices)

    # Compute theory prediction and reshape to
    # [n_correlations, n_ells]
    ind = unequal_enumerate(ls)

    tvd = eval_and_unwrap(params, thd.th, ind)
    tv1hd = eval_and_unwrap(params, thd.th1h, ind)
    tv2hd = eval_and_unwrap(params, thd.th2h, ind)

    tvg = eval_and_unwrap(params, thg.th, ind)
    tv1hg = eval_and_unwrap(params, thg.th1h, ind)
    tv2hg = eval_and_unwrap(params, thg.th2h, ind)

    # Reshape data vector
    dv = unwrap(lik.dv, ind)
    # Compute error bars and reshape
    ev = unwrap(np.sqrt(np.diag(lik.cv)), ind)


    for i in range(2):

        # set-up subplot
        gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[s, i],
                                     height_ratios=[3, 1], hspace=0)
        ax1 = f.add_subplot(gs[0])
        ax2 = f.add_subplot(gs[1])

        # Residuals and formatting plot
        res = (dv[i]-tvg[i])/ev[i]
        ax2.axhline(color="k", ls="--")
        ax2.errorbar(ls[i], res, yerr=np.ones_like(dv[i]), fmt="r.")

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.set_xscale("log")

        ax1.set_xlim(ls[i][0]/1.1, ls[i][-1]*1.1)
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_ylim(res.min()-1, res.max()+1)


        # plot data & theory
        ax1.plot(ls[i], tv1hg[i], ls=":", c="darkgreen", alpha=0.3)
        ax1.plot(ls[i], tv1hd[i], ls="-", c="darkgreen", alpha=0.3,
                 label=r"$\mathrm{1}$-$\mathrm{halo}$")

        ax1.plot(ls[i], tv2hg[i], ls=":", c="navy", alpha=0.3)
        ax1.plot(ls[i], tv2hd[i], ls="-", c="navy", alpha=0.3,
                 label=r"$\mathrm{2}$-$\mathrm{halo}$")

        ax1.plot(ls[i], tvg[i], ls=":", c="k")
        ax1.plot(ls[i], tvd[i], ls="-", c="k", label=r"$\mathrm{1h+2h}$")


        ax1.errorbar(ls[i], dv[i], yerr=ev[i], fmt="r.")


        # grey boundaries
        lmin = v["twopoints"][i]["lmin"]
        ax1.axvspan(ax1.get_xlim()[0], lmin, color="grey", alpha=0.2)
        ax2.axvspan(ax1.get_xlim()[0], lmin, color="grey", alpha=0.2)
        ax1.axvspan(lmax, ax1.get_xlim()[1], color="grey", alpha=0.2)
        ax2.axvspan(lmax, ax1.get_xlim()[1], color="grey", alpha=0.2)

        if i == 0:
            ax1.set_ylabel('$C_\\ell$', fontsize=15)
            ax2.set_ylabel('$\\Delta_\\ell$', fontsize=15)

        if s == 0:
            if i == 0:
                ax1.text(0.45, 1.1, r"$g \times g$", fontsize=15,
                         transform=ax1.transAxes)
            if i == 1:
                ax1.text(0.45, 1.1, r"$y \times g$", fontsize=15,
                         transform=ax1.transAxes)
                ax1.legend(loc="lower center", ncol=3, frameon=False,
                           bbox_to_anchor=(0.52, -0.1))

        if s != len(surveys)-1:
            ax2.get_xaxis().set_visible(False)
        else:
            ax2.set_xlabel('$\\ell$', fontsize=15)


f.tight_layout(h_pad=0.05, w_pad=0.1)
f.show()
#f.savefig("images/fits/fits_v3.pdf", bbox_inches="tight")