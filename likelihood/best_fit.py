import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl

import fittingtools as ft
import profile2D as p2D
import pspec as pspec



dir1 = "../analysis/data/dndz/"

# survey properties: name, zrange, bin
sprops = {"2mpz"     :  [(0.0005, 0.0745), 1],
          "wisc_b1"  :  [(0.0005, 0.1405), 1],
          "wisc_b2"  :  [(0.0215, 0.1575), 2],
          "wisc_b3"  :  [(0.0505, 0.1785), 3],
          "wisc_b4"  :  [(0.0735, 0.2035), 4],
          "wisc_b5"  :  [(0.0895, 0.2355), 5]}
surveys = list(sprops.keys())

cosmo = ccl.Cosmology(Omega_c=0.26066676, Omega_b=0.048974682, h=0.6766,
                      sigma8=0.8102, n_s=0.9665)


popt = np.array([np.load("fit_vals/"+sur+".npy") for sur in surveys])



## DATA ##
# outer --> inner: (l,cl,dcl) --> (yg,gg) --> (surveys)
l, cl, dcl = [[[[] for s, _ in enumerate(surveys)] for i in range(2)] for i in range(3)]
for i, sur in enumerate(surveys):
    data = [sur+","+"y_milca", sur+","+sur]
    l_arr, cl_arr, I, _ = ft.dataman(data, z_bin=sprops[sur][1], cosmo=cosmo)

    l[0][i], l[1][i] = l_arr
    cl[0][i], cl[1][i] = cl_arr
    dcl[0][i], dcl[1][i] = np.split(np.sqrt(np.diag(np.linalg.inv(I))), (len(l[0][i]),))



## MODEL ##
chi2, Np = [[[[] for s, _ in enumerate(surveys)] for i in range(2)] for i in range(2)]
Cl = [[[] for s, _ in enumerate(surveys)] for i in range(2)]
hal = [[[[] for s, _ in enumerate(surveys)] for i in range(2)] for i in range(2)]
p1 = p2D.Arnaud()
for i, sur in enumerate(surveys):
    params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc", "b_hydro"]
    kwargs = dict(zip(params, popt[i][0]))

    fname = dir1 + ("2MPZ_bin1.txt" if sur is "2mpz" else  sur[:4].upper() +
                    "_bin%d.txt" % sprops[sur][1])
    p2 = p2D.HOD(nz_file=fname)

    # yxg
    Cl[0][i] = pspec.ang_power_spectrum(cosmo, l[0][i], (p1, p2),
                                        zrange=sprops[sur][0], **kwargs)
    # gxg
    Cl[1][i] = pspec.ang_power_spectrum(cosmo, l[1][i], (p2, p2),
                                        zrange=sprops[sur][0], **kwargs)

    for j in range(2):
        if j == 1: p1 = p2
        hal[j][0][i] = pspec.ang_power_spectrum(cosmo, l[0][i], (p1, p2),
                                   zrange=sprops[sur][0],
                                   include_1h=True, include_2h=False, **kwargs)
        hal[j][1][i] = pspec.ang_power_spectrum(cosmo, l[0][i], (p1, p2),
                                   zrange=sprops[sur][0],
                                   include_1h=False, include_2h=True, **kwargs)

    # stats
    Np[0][i] = len(Cl[0][i])
    chi2[0][i] = np.sum((cl[0][i]-Cl[0][i])**2 / (dcl[0][i])**2)

    Np[1][i] = len(Cl[1][i])
    chi2[1][i] = np.sum((cl[1][i]-Cl[1][i])**2 / (dcl[1][i])**2)

    print(i+1, "/", len(surveys))



## PLOTS ##
# setup
nrows, ncols = 2, 3
F = [plt.subplots(nrows, ncols, sharex=True, sharey="row", figsize=(15,10),
                  gridspec_kw={"wspace":0.05, "hspace":0.05}) for i in range(2)]
F = [list(x) for x in F]
# axes labels
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

F[0][0].savefig("../images/best_fit_clyg.pdf", dpi=1000, bbox_inches="tight")
F[1][0].savefig("../images/best_fit_clgg.pdf", dpi=1000, bbox_inches="tight")