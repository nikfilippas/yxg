import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
from matplotlib.cm import copper



dir1 = "../analysis/data/dndz/"

# g-surveys
wisc = ["wisc_b%d" % i for i in range(1, 6)]
surveys = ["2mpz"] + wisc

# MCMC results: 50, 16, 84 quantiles
vals = [np.load("fit_vals/%s.npy" % sur) for sur in surveys]
bh = np.array([[val[i, 1] for val in vals] for i in range(3)])

z_avg = np.zeros((3, len(surveys)))
z, N = [[[] for j in surveys] for i in range(2)]
for i, s in enumerate(surveys):
    # data manipulation
    fname = dir1 + ("2MPZ_bin1.txt" if s is "2mpz" else  s[:4].upper() + "_bin%d.txt" % i)
    zd, Nd = np.loadtxt(fname, unpack=True)
    Nd /= simps(Nd, x=zd)  # normalise histogram
    #surveys
    z[i] = zd
    N[i] = Nd
    # average redshift & std
    z_avg[0, i] = np.average(zd, weights=Nd)
    perc = np.array([np.trapz(Nd[:j], zd[:j]) for j, _ in enumerate(zd)])
    # redshifts at 16, 84 percentiles
    z_avg[1, i], z_avg[2, i] = [np.abs(z_avg[0, i] - zd[np.abs(perc-p).argmin()]) for p in [0.16, 0.84]]

z, N = map(lambda x: [np.array(y) for y in x], [z, N])



# PLOT #
col = [copper(i) for i in np.linspace(0, 1, len(surveys))]

fig, (hist, ax) = plt.subplots(2, 1, sharex=True, figsize=(10, 10),
                         gridspec_kw={"height_ratios":[1, 3], "hspace":0.05})


ax.set_xlabel("$\mathrm{z}$", fontsize=15)
ax.set_ylabel("$\mathrm{b_{hydro}}$", fontsize=15)
hist.set_ylabel("$\mathrm{dN/dz}$", fontsize=15)

hist.set_xlim(0, 0.4)
[hist.fill_between((z_avg[0,i]-z_avg[1,i], z_avg[0,i]+z_avg[2,i]),
                   np.min(N)*np.ones_like(z_avg[0,i]), np.max(N)*np.ones_like(z_avg[0,i]),
                   color=col[i], alpha=0.15) for i, _ in enumerate(z_avg[0])]
hist.vlines(z_avg[0], np.min(N), np.max(N), colors="salmon", linestyles=":", alpha=0.7)

[hist.plot(z[i], N[i], c=col[i], label=surveys[i]) for i, _ in enumerate(surveys)]
markers, caps, bars = ax.errorbar(z_avg[0], bh[0],
                                  xerr=z_avg[1:], yerr=bh[1:], color="salmon", fmt="s",
                                  ecolor="k", capsize=2)

[bar.set_alpha(0.5) for bar in bars]
[cap.set_alpha(0.5) for cap in caps]

hist.legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncol=len(surveys), fontsize=12, fancybox=True)
fig.show()
fig.savefig("../images/b_hydro.pdf", bbox_inches="tight")