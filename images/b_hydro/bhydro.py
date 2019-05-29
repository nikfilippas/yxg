import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.cm import copper
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)



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


def get_z_Nz(version=""):
    """Get a version of dn/dz's."""
    dz, dN = [[[] for j in surveys] for i in range(2)]
    for i, s in enumerate(surveys):
        # data manipulation
        fname = dir1 + ("2MPZ" + version + "_bin1.txt" if s is "2mpz" \
                        else  s[:4].upper() + version + "_bin%d.txt" % i)
        zd, Nd = np.loadtxt(fname, unpack=True)
        Nd /= simps(Nd, x=zd)  # normalise histogram
        #surveys
        dz[i] = zd
        dN[i] = Nd

    dz, dN = map(lambda x: [np.array(y) for y in x], [dz, dN])
    return dz, dN


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
sci = [r"$\mathrm{2MPZ}$"] + \
      [r"$\mathrm{WI \times SC}$ - $\mathrm{%d}$" % i for i in range(1, 6)]

dz1, dN1 = get_z_Nz()
dz2, dN2 = get_z_Nz(version="_v2")
bins = np.array([np.argmax(np.array(dN1)[:, i]) for i, _ in enumerate(dz1[0])])
bounds = np.array([np.where(bins == i)[0][0] for i, _ in enumerate(dz1)])
zbounds = np.append(dz1[0][bounds], np.max(dz1))


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
lbls = ["fiducial", "$y$-NILC", "Tinker10",
        r"$\mathrm{d} n \mathrm{/d} z$", "high mass mask"]
col = [copper(i) for i in np.linspace(0, 1, len(surveys))]


fig, (hist, ax) = plt.subplots(2, 1, sharex=True, figsize=(10, 12),
                         gridspec_kw={"height_ratios":[1, 3], "hspace":0.05})


ax.axhline(0.58, ls=":", color="grey")
ax.axhspan(0.58-0.04, 0.58+0.06, color="grey", alpha=0.3)
ax.axhline(0.72, ls=":", color="cadetblue")
ax.axhspan(0.72-0.10, 0.72+0.10, color="cadetblue", alpha=0.3)
ax.annotate("CMB + cluster counts",
            (0.005, 0.585), fontsize=14, fontweight="bold")
ax.annotate("CMB lensing", (0.005, 0.725),
            fontsize=14, fontweight="bold")
ax.set_xlabel("$z$", fontsize=17)
ax.set_ylabel("$1-b_H$", fontsize=17)
hist.set_ylabel(r"$\mathrm{d} N \mathrm{/d} z$", fontsize=17)

for i, (a, c, fmt, lbl) in enumerate(zip(data, colours, fmts, lbls)):
    plotfunc(ax, a+0.003*i, xerr=None, fmt=fmt, color=c, label=lbl)

handles, labels = ax.get_legend_handles_labels()
ax.legend([object]+handles, ["Planck15"]+labels,
          handler_map={object: AnyObjectHandler()},
          loc="lower right", fontsize=14, ncol=2)

ax.set_xlim(0, 0.4)
#ax.set_xlim(ax.get_xlim())  # fix xlim
ax.tick_params(labelsize="large")

[hist.axvspan(zbounds[i], zbounds[i+1],
              color=col[i], alpha=0.3) for i, _ in enumerate(dz1)]
[hist.plot(dz1[i], dN1[i],
           c=col[i], lw=2, label=sci[i]) for i, _ in enumerate(surveys)]
[hist.plot(dz2[i], dN2[i], ls=":", alpha=0.6,
           c=col[i], lw=2) for i, _ in enumerate(surveys)]
hist.legend(loc="lower center", bbox_to_anchor=[0.5, -0.14],
            ncol=len(surveys), fontsize=9.5, frameon=False)

hist.set_ylim(0, hist.get_ylim()[1])
hist.tick_params(labelsize="large")

plt.savefig("bhydro.pdf")
