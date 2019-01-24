"""
test of Arnaud profile
"""


import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl

import profile2D
import pspec


cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, h=0.66, sigma8=0.79, n_s=0.81)

pa = profile2D.Arnaud(rrange=(1e-3,1e3), qpoints=1024)

larr = np.logspace(1, np.log10(2e4), 128)
cl_1h = pspec.ang_power_spectrum(cosmo, larr, pa, pa, include_2h=False)
cl_2h = pspec.ang_power_spectrum(cosmo, larr, pa, pa, include_1h=False)
cl_tt = cl_1h + cl_2h

const = 1e12*larr*(larr+1)/(2*np.pi)
plt.loglog(larr, cl_1h*const, "r-", label="1-halo")
plt.loglog(larr, cl_2h*const, "b--", label="2-halo")
plt.loglog(larr, cl_tt*const, "y-.", label="Total")
plt.xlabel("$\\ell$", fontsize=16)
plt.ylabel(r"$10^{12} \ell (\ell+1)\,C_\ell / 2\pi$",fontsize=16)
plt.ylim(5e-3, 2)
plt.legend(loc="lower right", fontsize=14)
#plt.show()
#plt.savefig("benchmarks/clyy_final.pdf", bbox_inches="tight")
