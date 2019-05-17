import numpy as np
import matplotlib.pyplot as plt

ymilca = np.load("ymilca.npy")
ynilc = np.load("ynilc.npy")
masked_ymilca = np.load("masked_ymilca.npy")
masked_nilc = np.load("masked_ynilc.npy")
mfunc_ymilca = np.load("mfunc_ymilca.npy")
mfunc_ynilc = np.load("mfunc_ynilc.npy")
dndz_ymilca = np.load("dndz_ymilca.npy")
dndz_ynilc = np.load("dndz_ynilc.npy")

def plotfunc(a, xerr=None, fmt=None, color=None, alpha=0, label=None):
    """Plots b_hydro data with error bars."""
    x, y = a[0: 2]
    if xerr: xerr = a[2]
    yerr = a[3:]
    plt.errorbar(x, y, xerr=xerr, yerr=yerr,
                 fmt=fmt, c=color, alpha=alpha, label=label)


data = [ymilca, ynilc,
        masked_ymilca, masked_nilc,
        mfunc_ymilca, mfunc_ynilc,
        dndz_ymilca, dndz_ynilc]
colours = ["k", "k",
           "salmon", "salmon",
           "royalblue", "royalblue",
           "forestgreen", "forestgreen"]
fmts = ["o", "^"]*4
alphas = [1, 0.5]*4
lbls = ["ymilca", "ynilc",
        "masked_ymilca", "masked_nilc",
        "mfunc_ymilca", "mfunc_ynilc",
        "dndz_ymilca", "dndz_ynilc"]

for a, c, fmt, alpha, lbl in zip(data, colours, fmts, alphas, lbls):
    plotfunc(a, xerr=None, fmt=fmt, color=c, alpha=alpha, label=lbl)

plt.legend(loc="lower right", fontsize=10, ncol=2)
plt.xlabel("z", fontsize=14)
plt.ylabel(r"$\mathrm{b_{hydro}}$", fontsize=14)
#plt.savefig("bhydro.pdf")