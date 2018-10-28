"""
This script contains useful tests for the codes, as well as example figures
illustrating the inner workings of some classes, functions, and methods.

Sections/Tests are separated by description in triple quotations.
Uncomment trailing code to run test.
"""

## ARNAUD RATIO ##
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import viridis as cmap
from matplotlib.cm import viridis_r as cmap_r
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import pyccl as ccl

import pspec
from profiles import Arnaud
from cosmotools import scale_factor

savedir = "images/"  # output directory


"""
ARNAUD RATIO
"""
#def Arnaud_ratio(x, M, z):
#    """Computes the ratio of two Arnaud profiles of a given mass and at a given
#    redshift. The profiles are identical, except one uses the full Arnaud profile
#    including a radial mass dependence.
#    """
#    r = Arnaud(x, M, z, aPP=False) / Arnaud(x, M, z, aPP=True)
#    return r
#
#
#
#x = np.logspace(-3, np.log10(2), 1000)  # R/R_Delta
#
#M_array = np.logspace(np.log10(6.0e13), np.log10(1.3e15), 8)
#M_dep = [Arnaud_ratio(x, M=M, z=0) for M in M_array]
#
#
#fig, ax = plt.subplots(1, 1, figsize=(5,5))
#ax.set_xlabel("$R \mathrm{ /R_{500} }$", fontsize=16)
#lbl = r"$\frac{ P_{th}\left( x, M_{500}, z \right) }{ P_{th}\left( x, M_{500}, z, \alpha(x) \right) }$"
#ax.set_ylabel(lbl, fontsize=16)
#
#ax.axhline(y=1, ls="--", c="k")
#
#colors = cmap(np.linspace(0, 1, len(M_array)))
#for i, _ in enumerate(M_array):
#    txt = r"$ %.1e\ \mathrm{M_{\odot}}$" % M_array[i]
#    txt = txt[:5] + r"\times 10^{%s}" % txt[7:9] + txt[10:]
#    ax.plot(x, M_dep[i], lw=2, c=colors[i], label=txt)
#
#ax.margins(x=0)
#ax.legend(loc="lower right", ncol=2, fontsize=10)
#fig.savefig(savedir+"arnaud_ratio.pdf", bbox_inches="tight")
#plt.close()



"""
3D SPACE OF "MASS", "WAVE VECTOR", "POWER SPECTRUM"
"""
## Fiducial Cosmology
#cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, A_s=2.1e-9, n_s=0.96)
#
#p1 = pspec.Profile(cosmo, "arnaud")
#p2 = pspec.Profile(cosmo, "arnaud")
#
## params
#k_arr = np.logspace(-1, 1, 100)
#z = 0
#
## Set up integration bounds
#logMmin, logMmax = 10, 16  # log of min and max halo mass [Msol]
#mpoints = 10  # number of integration points
#
#M_arr = np.logspace(logMmin, logMmax, mpoints)
#I = np.zeros((len(k_arr), len(M_arr)))  # initialize
#for m, M in enumerate(M_arr):
#    U = p1.fourier_profile(k_arr, M, z)
#    V = p2.fourier_profile(k_arr, M, z)
#
##    mfunc = ccl.massfunc(cosmo, M, scale_factor(z))
#    mfunc = 1e-4  # FIXME: replace with CCL (see above)
#
#    I[:, m] = mfunc*U*V
#
#
#plt.imshow(I.T,
#           origin="lower",
#           extent=[np.log10(k_arr.min()), np.log10(k_arr.max()),
#                   np.log(M_arr.min()), np.log10(M_arr.max())],
#           norm=LogNorm(vmin=I.min(), vmax = I.max()))
#plt.gca().invert_yaxis()
#
#plt.xlabel("k", fontsize=16)
#plt.ylabel("log(M)", fontsize=16)
#plt.savefig(savedir+"logM_vs_k.pdf", bbox_inches="tight")
#plt.close()



"""
2D graph of masses VS k-vectors
"""
#colors = cmap_r(np.linspace(0, 1, len(I[0])))
#_ = [plt.loglog(k_arr, I[:, i], c=colors[i]) for i, _ in enumerate(I[0])]
#plt.xlabel("k", fontsize=16)
#plt.ylabel("1-halo power spectrum contribution", fontsize=16)
#plt.savefig(savedir+"Puv1h_vs_k.pdf", bbox_inches="tight")
#plt.close()



"""
3D mesh of    P_UV^1h .. M .. k
"""
#fig = plt.figure()
#ax = fig.gca(projection="3d")
#
#X, Y = np.meshgrid(np.log10(k_arr), np.log10(M_arr))
#ax.plot_surface(X, Y, np.log10(I.T), cmap=cmap)
#
#ax.set_xlabel("log(k)", fontsize=16)
#ax.set_ylabel("log(M)", fontsize=16)
#ax.set_zlabel(r"$ \log{\left( P_{UV}^{1h} \right)} $", fontsize=16)
#
#plt.savefig(savedir+"Puv1h_vs_k_vs_M.pdf", bbox_inches="tight")
#plt.close()
