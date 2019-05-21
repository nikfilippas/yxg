import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase


class AnyObjectHandler(HandlerBase):
# Adapted from:
# https://matplotlib.org/users/legend_guide.html#legend-handlers
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height],
                                                linestyle="--", color="brown")
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height],
                                                linestyle="--", color="grey")
        return [l1, l2]


def plotfunc(ax, a, xerr=None, fmt=None, color=None, label=None):
    """Plots b_hydro data with error bars."""
    x, y = a[0: 2]
    if xerr: xerr = a[2]
    yerr = a[3:]
    ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                 fmt=fmt, c=color, label=label)


fiducial = np.load("fiducial.npy")
ynilc = np.load("ynilc.npy")
masked = np.load("masked.npy")
mfunc = np.load("mfunc.npy")
dndz = np.load("dndz.npy")

z = fiducial[0]  # probed redshifts


data = [fiducial, dndz, masked, mfunc, ynilc]
colours = ["k", "salmon", "royalblue", "forestgreen", "deeppink"]
fmts = ["o"]*len(data)
lbls = ["fiducial", "dndz", "masked", "mfunc", "ynilc"]


fig, ax = plt.subplots(figsize=(10, 7))

ax.axhline(0.58, ls="--", color="grey")
ax.axhspan(0.58-0.06, 0.58+0.06, color="grey", alpha=0.3)
ax.axhline(0.72, ls="--", color="brown")
ax.axhspan(0.72-0.10, 0.72+0.10, color="brown", alpha=0.3)

ax.annotate(r"\textbf{CMB + cluster counts}", (0.055, 0.585), fontsize=12)
ax.annotate(r"\textbf{CMB lensing}", (0.055, 0.725), fontsize=12)


for i, (a, c, fmt, lbl) in enumerate(zip(data, colours, fmts, lbls)):
    plotfunc(ax, a+0.005*i, xerr=None, fmt=fmt, color=c, label=lbl)


handles, labels = ax.get_legend_handles_labels()
ax.legend([object]+handles, ["Planck18"]+labels,
          handler_map={object: AnyObjectHandler()},
          loc="lower center", fontsize=14, ncol=2)

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

ax.set_xlabel(r"$z$", fontsize=16)
ax.set_ylabel(r"$1-b_H$", fontsize=16)
plt.savefig("bhydro.pdf")