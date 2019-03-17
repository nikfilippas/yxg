import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
import corner
import emcee

import fittingtools as ft
import profile2D as p2D
import pspec as pspec



# PARAMETER : [VALUE, STATUS, CONSTRAINTS] #
# (free : 0) ::: (fixed : 1) ::: (coupled : -N)
priors = {"Mmin"       :  [11.99,   -1,   (10, 16)],
          "M0"         :  [11.99,   -1,   (10, 16)],
          "M1"         :  [13.18,   0,   (10, 16)],
          "sigma_lnM"  :  [0.26,    1,   (0.1, 1.0)],
          "alpha"      :  [1.43,    1,   (0.5, 1.5)],
          "fc"         :  [0.54,    1,   (0.1, 1.0)],
          "bg"         :  [1.0,     1,   (0, np.inf)],
          "bmax"       :  [1.0,     1,   (0, np.inf)],
          "r_corr"     :  [0.0,     0,   (-1, 1)],
          "b_hydro"    :  [0.45,    0,   (0.1, 0.9)]}


# SURVEY PROPERTIES #
dir1 = "../analysis/data/dndz/"
wisc = ["wisc_b%d" % i for i in range(1, 6)]
surveys = ["2mpz"] + wisc
bins = np.append([1], np.arange(1, 6))
sprops = ft.survey_properties(dir1, surveys, bins)



# INPUT #
samplers = [emcee.backends.HDFBackend("samplers/%s" % sur) for sur in surveys]
#tau = [np.max(sampler.get_autocorr_time()) for sampler in samplers]
tau = np.zeros(6)
burnin = [int(np.ceil(2*np.max(t))) for t in tau]
print(burnin)

samples = [sampler.get_chain(discard=burn, flat=True) for sampler, burn in zip(samplers, burnin)]
popt = [np.median(sample, axis=0) for sample in samples]

free, fixed, coupled = ft.split_kwargs(**priors)

# parameters
cosmo = ccl.Cosmology(Omega_c=0.26066676,
                      Omega_b=0.048974682,
                      h=0.6766,
                      sigma8=0.8102,
                      n_s=0.9665)



# outer --> inner: (l,cl,dcl) --> (yg,gg) --> (surveys)
l, cl, dcl = [[[[] for s in surveys] for i in range(2)] for i in range(3)]
chi2, Np = [[[[] for s in surveys] for i in range(2)] for i in range(2)]
Cl = [[[] for s in surveys] for i in range(2)]
hal = [[[[] for s in surveys] for i in range(2)] for i in range(2)]
p1 = p2D.Arnaud()

for i, sur in enumerate(surveys):
    ## DATA ##
    data = [sur+","+"y_milca", sur+","+sur]
    l_arr, cl_arr, I, _, _ = ft.dataman(data, z_bin=sprops[sur][1], cosmo=cosmo)

    l[0][i], l[1][i] = l_arr
    cl[0][i], cl[1][i] = cl_arr
    dcl[0][i], dcl[1][i] = np.split(np.sqrt(np.diag(np.linalg.inv(I))), (len(l[0][i]),))


    ## MODEL ##
    kwargs = ft.build_kwargs(popt[i], free, fixed, coupled, full=False)

    fname = dir1 + sur.strip("_b%d" % i).upper() + "_bin%d.txt" % sprops[sur][1]
    p2 = p2D.HOD(nz_file=fname)

    # yxg
    Cl[0][i] = pspec.ang_power_spectrum(cosmo, l[0][i], (p1, p2), sprops[sur][0], **kwargs)
    hal[0][0][i] = pspec.ang_power_spectrum(cosmo, l[0][i], (p1, p2), sprops[sur][0], include_1h=True, include_2h=False, **kwargs)
    hal[0][1][i] = pspec.ang_power_spectrum(cosmo, l[0][i], (p1, p2), sprops[sur][0], include_1h=False, include_2h=True, **kwargs)

    # gxg
    Cl[1][i] = pspec.ang_power_spectrum(cosmo, l[1][i], (p2, p2), sprops[sur][0], **kwargs)
    hal[1][0][i] = pspec.ang_power_spectrum(cosmo, l[0][i], (p2, p2), sprops[sur][0], include_1h=True, include_2h=False, **kwargs)
    hal[1][1][i] = pspec.ang_power_spectrum(cosmo, l[0][i], (p2, p2), sprops[sur][0], include_1h=False, include_2h=True, **kwargs)

    # stats
    Np[0][i] = len(Cl[0][i])
    chi2[0][i] = np.sum((cl[0][i]-Cl[0][i])**2 / (dcl[0][i])**2)

    Np[1][i] = len(Cl[1][i])
    chi2[1][i] = np.sum((cl[1][i]-Cl[1][i])**2 / (dcl[1][i])**2)



## PLOTS ##
lbl = {"Mmin"       :  "$\\mathrm{M_{min}}$",
       "M0"         :  "$\\mathrm{M_0}$",
       "M1"         :  "$\\mathrm{M_1}$",
       "sigma_lnM"  :  "$\\mathrm{\\sigma_{\\ln M}}$",
       "alpha"      :  "$\\mathrm{\\alpha}$",
       "fc"         :  "$\\mathrm{f_c}$",
       "bg"         :  "$\\mathrm{b_{s,g}}$",
       "bmax"       :  "$\\mathrm{b_{max}}$",
       "r_corr"     :  "$\\mathrm{r_{corr}}$",
       "b_hydro"    :  "$\\mathrm{b_{hydro}}$"}

yax = [lbl[key] for key in free]
yax.append(*["$\\mathrm{=}$".join([lbl[key] for key in c]) for c in coupled])



# Figure 1 :: burn-in #
for s, _ in enumerate(sprops):
    dims = samplers[s].shape
    fig, ax = plt.subplots(dims[1], 1, sharex=True, figsize=(7, 12))
    ax[-1].set_xlabel("step number", fontsize=15)
    for i in range(dims[1]):
        for j in range(dims[0]):
            ax[i].plot(samplers[s].get_chain()[:, j, i], "k-", lw=0.5, alpha=0.05)
        ax[i].get_yaxis().get_major_formatter().set_useOffset(False)
        ax[i].set_ylabel(yax[i], fontsize=15)
    plt.tight_layout()
    fig.show()
    sur = list(sprops.keys())[s]
    fig.savefig("../images/MCMC/MCMC_steps_%s.pdf" % sur, bbox_inches="tight")
    fig.close()


# Figure 2 :: corner plot #
for sample, sur in zip(samples, list(sprops.keys())):
    fig = corner.corner(sample, labels=yax, quantiles=[0.16, 0.50, 0.84],
                        show_titles=True, label_kwargs={"fontsize":15})
    fig.show()
    fig.savefig("../images/MCMC/MCMC_%s.pdf" % sur, bbox_inches="tight")



# Figure 3 :: best-fits #
nrows, ncols = 2, 3
F = [plt.subplots(nrows, ncols, sharex=True, sharey="row", figsize=(15,10),
                  gridspec_kw={"wspace":0.05, "hspace":0.05}) for i in range(2)]
F = [list(x) for x in F]
for i, row in enumerate(np.column_stack((F[0][1], F[1][1]))):
    for j, ax in enumerate(row):
        if i == nrows-1: [a.set_xlabel("$\\ell$", fontsize=15) for a in row]
        if j == 0: ax.set_ylabel("$C^{yg}_{\\ell}$", fontsize=15)
        if j == ncols: ax.set_ylabel("$C^{gg}_{\\ell}$", fontsize=15)

# plot
F[0][1], F[1][1] = map(lambda x: x.flatten(), [F[0][1], F[1][1]])
[[x.loglog() for x in y] for y in [F[0][1], F[1][1]]]
col = ["forestgreen", "royalblue"]
for i, sur in enumerate(surveys):

    for j, _ in enumerate(F):
        # data #
        F[j][1][i].errorbar(l[j][i], cl[j][i], dcl[j][i], fmt="o", color="salmon",
                            lw=3, ecolor="grey", elinewidth=2)
        # model #
        # 1h, 2h (loop works because len(F) == len(halos))
        F[j][1][i].plot(l[j][i], Cl[j][i], c="crimson", lw=3)
        for k in range(2):
            F[k][1][i].plot(l[j][i], hal[k][j][i], c=col[j], lw=1.5)

        txt = "%s \n $\\mathrm{\\chi^2=%d}$ \n $\\mathrm{N=%d}$" % \
                (sur.upper(), chi2[j][i], Np[j][i])
        F[j][1][i].text(0.84, 0.80, txt, transform=F[j][1][i].transAxes, ha="center",
                      bbox={"edgecolor":"w", "facecolor":"white", "alpha":0}, fontsize=14)

F[0][0].show()
F[1][0].show()
#    F[0][0].savefig("../images/best_fit_clyg.pdf", dpi=1000, bbox_inches="tight")
#    F[1][0].savefig("../images/best_fit_clgg.pdf", dpi=1000, bbox_inches="tight")



# best fit values
for s, sur in enumerate(sprops.keys()):
    val = list(map(lambda v: (v[1], v[1]-v[0], v[2]-v[1]),
                   zip(*np.percentile(samples[s], [16, 50, 84], axis=0))))
    np.save("fit_vals/"+sur, np.array(val).T)