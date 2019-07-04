import os
#os.chdir("../../")
import numpy as np
from likelihood.chanal import chan
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.cm import bone
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True


def covplot(cov, save=None):
    lt = 100*np.min(np.abs(cov))
    q = SymLogNorm(linthresh=lt, vmin=cov.min(), vmax=cov.max())
    sci = [r"$\mathrm{2MPZ}$"] + \
          [r"$\mathrm{WISC}$ $\mathrm{%d}$" % i for i in range(1, 6)]

    fig, ax = plt.subplots()
    cp = ax.imshow(cov, cmap=bone, norm=q)
    fig.colorbar(cp)

    plt.xticks(np.arange(6), sci, fontsize=8)
    plt.yticks(np.arange(6)-0.25, sci, rotation=90, fontsize=8)

    plt.setp(ax.get_xticklines(), visible=False)
    plt.setp(ax.get_yticklines(), visible=False)

    if save is not None:
        os.chdir("images/covar/")
        fig.savefig(save, bbox_inches="tight")
        os.chdir("../../")


bh = np.vstack([np.load("output_default/b_hydro_jackknife%d.npy" % jk)
                 for jk in np.arange(1, 461)])

bh_cov = (len(bh)-1)*np.cov(bh.T, bias=True)
np.save("output_default/bhydro_covar.npy", bh_cov)
covplot(bh_cov, save="bhydro_covar.pdf")



## extra work for b_y covariance ##
dd = chan("params_default.yml", error_type="hpercentile", chains=False, b_hydro=bh)
by = np.column_stack([x["by"] for x in dd[0]])

by_cov = (len(by)-1)*np.cov(by.T, bias=True)
np.save("output_default/bhydro_covar.npy", bh_cov)
covplot(by_cov, save="by_covar.pdf")
