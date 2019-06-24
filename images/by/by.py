import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

run_name = "lmin10_kmax1_tinker08_ymilca"
data = np.load("../../output_default/"+run_name+"_bH.npy")

z = data[0]
by = data[-3]
sby = data[-2:]

DESx = np.array([0.14, 0.23, 0.24, 0.39, 0.40, 0.52, 0.53, 0.67, 0.68])
DESy = 1e-1*np.array([1.4, 1.4, 0.9, 2.4, 2.5, 3.7, 3.0, 2.5, 2.2])
DESsy = 1e-1*np.array([0.15, 0.50, 0.65, 0.60, 0.50, 0.80, 0.50, 1.20, 0.75])

DES = np.vstack((DESx, DESy, DESsy))

black = DES[:, 0]
green = DES[:, 1::2]
orang = DES[:, 2::2]

fig, ax = plt.subplots(figsize=(9,7))
ax.errorbar(z, 1e3*by, 1e3*sby, fmt="o", c="royalblue", elinewidth=2, label="this work")

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
fig.savefig("by.pdf", bbox_inches="tight")