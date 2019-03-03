import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
import pyccl as ccl

import profile2D
import pspec



## INPUT ##
profplot = "yg"
dir1 = "../../analysis/data/dndz/"

# g-surveys
wisc = ["wisc_b%d" % i for i in range(1, 6)]
surveys = ["2mpz"] + wisc


# range selection #
Z = [[] for i, _ in enumerate(surveys)]
for i, sur in enumerate(surveys):
    k = 1 if sur is "2mpz" else i
    fname = dir1 + sur.strip("_b%d" % i).upper() + "_bin%d.txt" % k
    z, N = np.loadtxt(fname, unpack=True)
    m = N.argmax()
    imin = np.abs(N[:m] - N.max()/100).argmin()
    imax = np.abs(N[m:] - N.max()/100).argmin()
    Z[i] = (z[imin], z[imax])


# parameters
l_arr = np.arange(6, 300)
cosmo = ccl.Cosmology(Omega_c=0.26066676, Omega_b=0.048974682, h=0.6766,
                      sigma8=0.8102, n_s=0.9665)
params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc", "b_hydro"]
popt = [11.99, 14.94, 13.18, 0.26, 1.43, 0.54, 0.45]
kwargs = dict(zip(params, popt))
sprops = {"2mpz"   :  [Z[0], 1],
          "wisc_b1":  [Z[1], 1],
          "wisc_b2":  [Z[2], 2],
          "wisc_b3":  [Z[3], 3],
          "wisc_b4":  [Z[4], 4],
          "wisc_b5":  [Z[5], 5]}



## BENCHMARK ##
prof1 = profile2D.Arnaud()
npoints = [32, 2048]
Cl = np.zeros((len(surveys), len(npoints), len(l_arr)))
for s, sur in enumerate(surveys):
    fname = dir1 + ("2MPZ_bin1.txt" if sur is "2mpz" else  sur[:4].upper() +
                    "_bin%d.txt" % sprops[sur][1])
    prof2 = profile2D.HOD(nz_file=fname)
    if profplot is "gg": prof1 = prof2
    for i, n in enumerate(npoints):
        zrange = sprops[sur][0], (0.1*sprops[sur][0][0], sprops[sur][0][1])
        Cl[s, i] = pspec.ang_power_spectrum(cosmo, l_arr, (prof1, prof2),
                                          zrange=zrange[i], zpoints=n,
                                          is_zlog=True, **kwargs)

    print(s+1,"/", len(surveys), sur)



## PLOTS ##
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ratio = np.abs(1- Cl[:, :-1] / Cl[:, 1:])

nrows, ncols = 2, 3
fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey="row",
                         figsize=(12,10), gridspec_kw={"wspace":0.05, "hspace":0.05})
fig.suptitle("$C^{%s}_{\\ell}$" % profplot, fontsize=18)
for i, row in enumerate(axes):
    for j, ax in enumerate(row):
        if i == nrows-1: ax.set_xlabel("$\\ell$", fontsize=15)
        if j == 0: ax.set_ylabel("$\\| 1 - \\frac{N_{32}}{N_{2048}} \\|$",
                       fontsize=15)
        ax.hlines(0.01, 0, 285, linestyle=":")

axes = axes.flatten()
for i, ax in enumerate(axes):
    ax.loglog(l_arr, ratio[i, 0, :], "g-", lw=3, label="%s T" % surveys[i])
    ax.legend(loc="lower right")

    ax.xaxis.set_major_formatter(ScalarFormatter())

fig.show()
fig.savefig("zrange_%s.pdf" % profplot, dpi=1000, bbox_inches="tight")