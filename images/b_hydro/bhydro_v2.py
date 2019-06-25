import os
os.chdir("../../")
import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
from analysis.params import ParamRun
from likelihood.chanal import chan
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


def plotfunc(ax, a, xerr=None, fmt=None, color=None, label=None):
    """Plots b_hydro data with error bars."""
    x, y = a[0: 2]
    if xerr: xerr = a[2]
    yerr = a[3:5]
    ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                 fmt=fmt, c=color, label=label)



def get_dndz(fname, width):
    """Get the modified galaxy number counts."""
    zd, Nd = np.loadtxt(fname, unpack=True)

    Nd /= simps(Nd, x=zd)
    zavg = np.average(zd, weights=Nd)
    nzf = interp1d(zd, Nd, kind="cubic", bounds_error=False, fill_value=0)

    Nd_new = nzf(zavg + (1/width)*(zd-zavg))
    return zd, Nd_new


# dn/dz
dir1 = "data/dndz/"

wisc = ["wisc_b%d" % i for i in range(1, 6)]
surveys = ["2mpz"] + wisc
sci = [r"$\mathrm{2MPZ}$"] + \
      [r"$\mathrm{WI \times SC}$ - $\mathrm{%d}$" % i for i in range(1, 6)]


bins = np.array([np.argmax(np.array(dN1)[:, i]) for i, _ in enumerate(dz1[0])])
bounds = np.array([np.where(bins == i)[0][0] for i, _ in enumerate(dz1)])
zbounds = np.append(dz1[0][bounds], np.max(dz1))


# b-hydro
data = []

param_yml = ["params_default.yml",
             "params_ynilc.yml",
             "params_tinker.yml",
             "params_kmax.yml"
             "params_masked.yml"]
for run in param_yml:
    p = ParamRun(run)
    pars, _ = chan(run)

    i = 0
    for g in p.get("maps"):
        if g["type"] == "g":
            width = pars[i]["width"]
            dz, dN = get_dndz(g["dndz"], width)

            i += 1


    pass


z = data[0][0]  # probed redshifts


# Plot
colours = ["k", "brown", "darkorange", "orangered", "y"]
fmts = ["o"]*len(data)
lbls = ["fiducial", "$y$-NILC", "Tinker10", r"$k_{max}$", "high mass mask"]
col = [copper(i) for i in np.linspace(0, 1, len(surveys))]


fig, (hist, ax) = plt.subplots(2, 1, sharex=True, figsize=(10, 12),
                         gridspec_kw={"height_ratios":[1, 3], "hspace":0.05})


ax.axhline(0.58, ls=":", color="grey")
ax.axhspan(0.58-0.04, 0.58+0.06, color="grey", alpha=0.3)
ax.axhline(0.72, ls=":", color="cadetblue")
ax.axhspan(0.72-0.10, 0.72+0.10, color="cadetblue", alpha=0.3)
props = dict(boxstyle="round", facecolor="w", alpha=0.2)
ax.text(0.005, 0.600, "CMB + cluster counts",
        fontsize=12, fontweight="bold", bbox=props)
ax.text(0.005, 0.740, "CMB lens. + cluster counts",
        fontsize=12, fontweight="bold", bbox=props)
ax.set_xlabel("$z$", fontsize=17)
ax.set_ylabel("$1-b_H$", fontsize=17)
hist.set_ylabel(r"$\mathrm{d} n \mathrm{/d} z$", fontsize=17)

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
hist.legend(loc="lower center", bbox_to_anchor=[0.5, -0.16],
            ncol=len(surveys), fontsize=9.5, frameon=False)

hist.set_ylim(0, hist.get_ylim()[1])
hist.tick_params(labelsize="large")

os.chdir("images/b_hydro/")
#plt.savefig("bhydro.pdf", bbox_inches="tight")