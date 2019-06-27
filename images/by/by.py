import os
os.chdir("../../")
import numpy as np
from likelihood.chanal import chan
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

fname_params = "params_wnarrow.yml"
pars, _, chains = chan(fname_params, diff=True, error_type="hpercentile")
bys = [bb for bb in 1e3*np.array(chains[1])]

z = np.array([par["z"] for par in pars])
by = np.array([par["by"] for par in pars]).T

DESx = np.array([0.14, 0.23, 0.24, 0.39, 0.40, 0.52, 0.53, 0.67, 0.68])
DESy = 1e-1*np.array([1.4, 1.4, 0.9, 2.4, 2.5, 3.7, 3.0, 2.5, 2.2])
DESsy = 1e-1*np.array([0.15, 0.50, 0.65, 0.60, 0.50, 0.80, 0.50, 1.20, 0.75])

DES = np.vstack((DESx, DESy, DESsy))

black = DES[:, 0]
green = DES[:, 1::2]
orang = DES[:, 2::2]

fig, ax = plt.subplots(figsize=(9,7))
ax.violinplot(bys, z, widths=0.02, showmedians=True, showextrema=False)
ax.errorbar(z, 1e3*by[0], 1e3*by[1:],
            fmt="o", c="royalblue", elinewidth=2, label="this work")

ax.errorbar(black[0], black[1], black[2], fmt="ko", elinewidth=2,
            label="Vikram16")
ax.errorbar(green[0], green[1], green[2], fmt="o", c="mediumseagreen", elinewidth=2,
            label="Pandey19: fiducial $y$ map")
ax.errorbar(orang[0], orang[1], orang[2], fmt="o", c="orangered", elinewidth=2,
            label="Pandey19: Planck NILC $y$ map")

ax.set_ylim(0,)

ax.tick_params(labelsize="large")
ax.set_xlabel("$z$", fontsize=17)
ax.set_ylabel(r"$\mathrm{\langle bP_e \rangle \ \big[ eV \ cm^{-3} \big] }$", fontsize=17)
ax.legend(loc="upper left", frameon=False, fontsize=14)

os.chdir("images/by/")
#fig.savefig("by.pdf", bbox_inches="tight")
