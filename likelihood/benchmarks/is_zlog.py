"""
testing convergence for is_log=True/False
"""

import sys
import os
sys.path.append(os.path.abspath(".."))

import numpy as np
import pyccl as ccl

import profile2D
import pspec


l_arr = np.arange(260)
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)

nz_file = "../../analysis/data/dndz/WISC_bin5.txt"
prof1 = profile2D.HOD(nz_file=nz_file)
z, nz = np.loadtxt(nz_file, unpack=True)

#prof2 = prof1
prof2 = profile2D.Arnaud()

kwargs={"Mmin"      : 12.3,
        "M0"        : 12.5,
        "M1"        : 13.65,
        "sigma_lnM" : 0.5,
        "alpha"     : 1.1,
        "fc"        : 0.8,
        "b_hydro"   : 0.4}


imin, imax = 5, 11
npoints = np.geomspace(2**imin, 2**imax, imax-imin+1)

Cl = [np.zeros((len(npoints), len(l_arr))) for i in range(2)]
for i, n in enumerate(npoints):
    Cl[0][i] = pspec.ang_power_spectrum(cosmo, l_arr, (prof1, prof1),
                                      zrange=(0.05, 0.6), zpoints=n,
                                      is_zlog=True, **kwargs)

    Cl[1][i] = pspec.ang_power_spectrum(cosmo, l_arr, (prof1, prof1),
                                      zrange=(0.05, 0.6), zpoints=n,
                                      is_zlog=False, **kwargs)

    print(n)



# PLOTS #
import matplotlib.pyplot as plt

ratio = [Cl[i][1:]/Cl[i][:-1] for i in range(2)]
q = 4
plt.subplots(1, 1)
plt.plot(l_arr, ratio[0][q], "k-", lw=3, label="$x=%d$, $\\mathtt{is\_log=True}$" % 2**(imin+q))
plt.plot(l_arr, ratio[0][q+1], "k-", lw=1, label="$x=%d$, $\\mathtt{is\_log=True}$" % 2**(imin+q+1))
plt.plot(l_arr, ratio[1][q], "r-", lw=3, label="$x=%d$, $\\mathtt{is\_log=False}$" % 2**(imin+q))
plt.plot(l_arr, ratio[1][q+1], "r-", lw=1, label="$x=%d$, $\\mathtt{is\_log=False}$" % 2**(imin+q+1))

plt.xlabel("$\\ell$", fontsize=15)
plt.ylabel("$\\mathrm{R_{2^{x+1}:2^x}}$", fontsize=15)
plt.legend(loc="lower right")
plt.show()


from matplotlib.cm import autumn, winter

col1 = [autumn(i) for i in np.linspace(0, 1, imax-imin+1)]
col2 = [winter(i) for i in np.linspace(0, 1, imax-imin+1)]

plt.subplots(1, 1)
plt.xlabel("$\\ell$", fontsize=15)
plt.ylabel("$C_{\\ell}$", fontsize=15)
[plt.loglog(l_arr, Cl[0][i], c=col1[i]) for i in range(imax-imin+1)]
[plt.loglog(l_arr, Cl[1][i], c=col2[i]) for i in range(imax-imin+1)]
plt.show()
#plt.savefig("test3.png", dpi=600)
