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

DESx = np.array([0.15, 0.24, 0.2495, 0.383, 0.393, 0.526, 0.536, 0.678, 0.688])
DESy = 1e-1*np.array([1.5, 1.51, 0.91, 2.46, 2.55, 3.85, 3.08, 2.61, 2.25])
DESsy_min = 1e-1*np.array([1.275, 0.940, 0.2587, 1.88, 2.092, 2.961, 2.377, 1.442, 1.284])
DESsy_max = 1e-1*np.array([1.726, 2.029, 1.593, 3.039, 2.991, 4.628, 3.620, 3.971, 2.994])

DESsy = np.vstack((DESy-DESsy_min, DESsy_max-DESy))

DES = np.vstack((DESx, DESy, DESsy))

black = DES[:, 0]
green = DES[:, 1::2]
orang = DES[:, 2::2]

fig, ax = plt.subplots(figsize=(9,7))
ax.violinplot(bys, z, widths=0.03, showextrema=False)
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