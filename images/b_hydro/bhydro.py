import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.cm import copper



class AnyObjectHandler(HandlerBase):
# Adapted from:
# https://matplotlib.org/users/legend_guide.html#legend-handlers
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                        lw=3, linestyle="-", color="cadetblue")
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height],
                        lw=3, linestyle="-", color="grey")
        return [l1, l2]


def plotfunc(ax, a, xerr=None, fmt=None, color=None, label=None):
    """Plots b_hydro data with error bars."""
    x, y = a[0: 2]
    if xerr: xerr = a[2]
    yerr = a[3:]
    ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                 fmt=fmt, c=color, label=label)


# dn/dz
dir1 = "../../data/dndz/"

wisc = ["wisc_b%d" % i for i in range(1, 6)]
surveys = ["2mpz"] + wisc

dz, dN = [[[] for j in surveys] for i in range(2)]
for i, s in enumerate(surveys):
    # data manipulation
    fname = dir1 + ("2MPZ_bin1.txt" if s is "2mpz"
                    else  s[:4].upper() + "_bin%d.txt" % i)
    zd, Nd = np.loadtxt(fname, unpack=True)
    Nd /= simps(Nd, x=zd)  # normalise histogram
    #surveys
    dz[i] = zd
    dN[i] = Nd

dz, dN = map(lambda x: [np.array(y) for y in x], [dz, dN])

bins = np.array([np.argmax(np.array(dN)[:, i]) for i, _ in enumerate(dz[0])])
bounds = np.array([np.where(bins == i)[0][0] for i, _ in enumerate(dz)])
zbounds = np.append(dz[0][bounds], np.max(dz))


# b-hydro
fiducial = np.load("fiducial.npy")
ynilc = np.load("ynilc.npy")
masked = np.load("masked.npy")
mfunc = np.load("mfunc.npy")
dndz = np.load("dndz.npy")

z = fiducial[0]  # probed redshifts



# Plot
data = [fiducial, ynilc, mfunc, dndz, masked]
colours = ["k", "brown", "darkorange", "orangered", "y"]
fmts = ["o"]*len(data)
lbls = ["fiducial", "y-nilc", "Tinker10", "dn/dz", "high mass mask"]
col = [copper(i) for i in np.linspace(0, 1, len(surveys))]


fig, (hist, ax) = plt.subplots(2, 1, sharex=True, figsize=(10, 12),
                         gridspec_kw={"height_ratios":[1, 3], "hspace":0.05})

ax.axhline(0.58, ls=":", color="grey")
ax.axhspan(0.58-0.06, 0.58+0.06, color="grey", alpha=0.3)
ax.axhline(0.72, ls=":", color="cadetblue")
ax.axhspan(0.72-0.10, 0.72+0.10, color="cadetblue", alpha=0.3)
ax.annotate("CMB + cluster counts",
            (0.055, 0.585), fontsize=9.5, fontweight="bold")
ax.annotate("CMB lensing", (0.055, 0.725),
            fontsize=9.5, fontweight="bold")
ax.set_xlabel(r"$\mathrm{z}$", fontsize=16)
ax.set_ylabel(r"$\mathrm{1-b_H}$", fontsize=16)
hist.set_ylabel("$\mathrm{dN/dz}$", fontsize=15)

for i, (a, c, fmt, lbl) in enumerate(zip(data, colours, fmts, lbls)):
    plotfunc(ax, a+0.003*i, xerr=None, fmt=fmt, color=c, label=lbl)

handles, labels = ax.get_legend_handles_labels()
ax.legend([object]+handles, ["Planck18"]+labels,
          handler_map={object: AnyObjectHandler()},
          loc="lower center", fontsize=14, ncol=3)

ax.set_xlim(ax.get_xlim())  # fix xlim

[hist.axvspan(zbounds[i], zbounds[i+1],
              color=col[i], alpha=0.3) for i, _ in enumerate(dz)]
[hist.plot(dz[i], dN[i],
           c=col[i], lw=2, label=surveys[i]) for i, _ in enumerate(surveys)]
hist.legend(loc="lower center", bbox_to_anchor=[0.5, 1],
            ncol=len(surveys), fontsize=11.2, fancybox=True)

plt.savefig("bhydro.pdf")