import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt

cosmo08 = ccl.Cosmology(Omega_c=0.26066676, Omega_b=0.048974682, h=0.6766, sigma8=0.8102, n_s=0.9665, mass_function="tinker")
cosmo10 = ccl.Cosmology(Omega_c=0.26066676, Omega_b=0.048974682, h=0.6766, sigma8=0.8102, n_s=0.9665, mass_function="tinker10")

M = np.logspace(11, 14, 7)
z = np.linspace(0.001, 0.5, 1000)
a = 1/(1+z)


mfunc08, mfunc10 = [[[]]*len(a) for i in range(2)]
for i, dist in enumerate(a):
    mfunc08[i] = list(ccl.massfunc(cosmo08, M, dist))
    mfunc10[i] = list(ccl.massfunc(cosmo10, M, dist))


mfunc08 = np.array(mfunc08).T
mfunc10 = np.array(mfunc10).T


alphas = np.linspace(1, 0.2, len(M))

fig, ax1 = plt.subplots()
for i, _ in enumerate(M):
    ax1.loglog(z, mfunc08[i], "r", lw=3, alpha=alphas[i])
    ax1.loglog(z, mfunc10[i], "b", lw=3, alpha=alphas[i])

plt.ylim(1e-5,)
plt.legend(["Tinker10", "Tinker08"], ncol=2, loc="lower center", fontsize=12)
plt.xlabel("z", fontsize=14)
plt.ylabel("$dn/dM$", fontsize=14)
plt.savefig("dndM.pdf")
##############################################################################

#from matplotlib.colors import Normalize

fig, ax2 = plt.subplots()

mdiff = np.abs(mfunc10 - mfunc08)
for i, _ in enumerate(M):
    ax2.loglog(z, mdiff[i], "k", lw=3, alpha=alphas[i])

#cmap = plt.cm.Greys
#norm = Normalize(vmin=11, vmax=14)
#
#
#cb = plt.colorbar(ax, cmap=cmap, norm=norm)
plt.xlabel("z", fontsize=14)
plt.ylabel("|Tinker10 - Tinker08|", fontsize=14)
plt.savefig("dndM_diff.pdf")