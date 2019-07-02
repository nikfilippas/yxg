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


def plotfunc(ax, zz, dd, inverted=False,
             fmt=None, color=None, label=None, offset=0):
    """Plots b_hydro data with error bars."""
    yerr = dd[1:] if not inverted else np.flip(dd[1:], axis=0)
    bh = dd[0] if not inverted else 1-dd[0]

    ax.errorbar(zz+0.004*offset, bh, yerr=yerr, fmt=fmt, c=color, label=label)



def get_dndz(fname, width):
    """Get the modified galaxy number counts."""
    zd, Nd = np.loadtxt(fname, unpack=True)

    Nd /= simps(Nd, x=zd)
    zavg = np.average(zd, weights=Nd)
    nzf = interp1d(zd, Nd, kind="cubic", bounds_error=False, fill_value=0)

    Nd_new = nzf(zavg + (1/width)*(zd-zavg))
    return zd, Nd_new



param_yml = ["params_wnarrow.yml",
             "params_ynilc.yml",
             "params_tinker.yml",
             "params_kmax.yml",
             "params_masked.yml"]
#param_yml = ["params_wfixed.yml",
#             "params_wnarrow.yml",
#             "params_wbroad.yml"]

sci = [r"$\mathrm{2MPZ}$"] + \
      [r"$\mathrm{WI \times SC}$ - $\mathrm{%d}$" % i for i in range(1, 6)]
lbls = ["fiducial", "$y$-NILC", "Tinker10", r"$k_{max}$", "high mass mask"]
#lbls = ["$w = 1$", r"$w \in [0.8, 1.2]$", r"$w \in [0.2, 2.0]$"]
colours = ["k", "brown", "darkorange", "orangered", "y"]
col = [copper(i) for i in np.linspace(0, 1, len(sci))]
fmts = ["o"]*len(param_yml)



p = ParamRun(param_yml[0])
temp = [chan(paryml, diff=True, error_type="hpercentile", chains=False)
         for paryml in param_yml]
pars = [t[0] for t in temp]
data = np.array([[p["b_hydro"] for p in par] for par in pars])
data = [d.T for d in data]


dz, dN = [[] for i in range(2)]
i = 0
for g in p.get("maps"):
    if g["type"] == "g":
        w = pars[0][i]["width"]  # [default][z-bins]["width"]
        w = w if type(w) is float else w[0]  # for fixed w
        zz, NN = get_dndz(g["dndz"], w)
        dz.append(zz)
        dN.append(NN)
        i += 1  # g-counter

z = np.array([np.average(zz, weights=NN) for zz, NN in zip(dz, dN)])

# find where probability of next bin > probability of previous bin,
# past the mode of the previous bin
mb = [np.argmax(dN[i]) for i, _ in enumerate(dz)]  # bin avg
bins = np.array([np.where((dN[i] < dN[i+1])[mb[i]:])[0][0] for i in range(len(dz)-1)])
bins += np.array(mb[:-1])
zbounds = np.append(np.append(0, dz[0][bins]), dz[0].max())  # add outer bins


# Plot
fig, (hist, ax) = plt.subplots(2, 1, sharex=True, figsize=(10, 12),
                         gridspec_kw={"height_ratios":[1, 3], "hspace":0.05})

ax.set_xlim(0, 0.4)
ax.tick_params(labelsize="large")
hist.set_ylim(0, np.ceil(np.max(dN)))
hist.tick_params(labelsize="large")

ax.axhline(0.58, ls=":", color="grey")
ax.axhspan(0.58-0.04, 0.58+0.06, color="grey", alpha=0.3)
ax.axhline(0.72, ls=":", color="cadetblue")
ax.axhspan(0.72-0.10, 0.72+0.10, color="cadetblue", alpha=0.3)
ax.axhline(0.44, ls=":", color="slateblue")
ax.axhspan(0.44-0.03, 0.44+0.03, color="slateblue", alpha=0.3)

props = dict(boxstyle="round", facecolor="w", alpha=0.2)
ax.text(0.005, 0.595, "CMB + cluster counts",
        fontsize=12, fontweight="bold", bbox=props)
ax.text(0.005, 0.735, "CMB lens. + cluster counts",
        fontsize=12, fontweight="bold", bbox=props)
ax.text(0.005, 0.460, "constant best-fit",
        fontsize=12, fontweight="bold", bbox=props)

ax.set_xlabel("$z$", fontsize=17)
ax.set_ylabel("$1-b_H$", fontsize=17)
hist.set_ylabel(r"$\mathrm{d} n \mathrm{/d} z$", fontsize=17)

for i, (dd, cc, fmt, lbl) in enumerate(zip(data, colours, fmts, lbls)):
    plotfunc(ax, z, dd, fmt=fmt, color=cc, label=lbl, inverted=True, offset=i)

handles, labels = ax.get_legend_handles_labels()
ax.legend([object]+handles, ["Planck15"]+labels,
          handler_map={object: AnyObjectHandler()},
          loc="lower right", fontsize=14, ncol=2)


[hist.axvspan(zbounds[i], zbounds[i+1],
              color=col[i], alpha=0.3) for i, _ in enumerate(dz)]
[hist.plot(dz[i], dN[i],
           c=col[i], lw=2, label=sci[i]) for i, _ in enumerate(sci)]
hist.legend(loc="lower center", bbox_to_anchor=[0.5, -0.15],
            ncol=len(sci), fontsize=9.5, frameon=False)


os.chdir("images/b_hydro/")
plt.savefig("bhydro.pdf", bbox_inches="tight")