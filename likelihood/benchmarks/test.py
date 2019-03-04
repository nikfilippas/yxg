import numpy as np
import pyccl as ccl
import matplotlib.pyplot as plt

import pspec as pspec
import profile2D as p2D


cosmo = ccl.Cosmology(Omega_c=0.26066676, Omega_b=0.048974682, h=0.6766, sigma8=0.8102, n_s=0.9665)
l = np.geomspace(6, 1300, 1000)

fname = "../analysis/data/dndz/2MPZ_bin1.txt"

#p1 = p2D.Arnaud()
p1 = p2D.HOD(nz_file=fname)
p2 = p2D.HOD(nz_file=fname)
profiles = (p1, p2)

params = ["Mmin", "M0", "M1", "sigma_lnM", "alpha", "fc", "b_hydro"]
popt = [11.99, 14.94, 13.18, 0.26, 1.43, 0.54, 0.45]
kwargs = dict(zip(params, popt))

Cl = pspec.ang_power_spectrum(cosmo, l, profiles, **kwargs)
C1h = pspec.ang_power_spectrum(cosmo, l, profiles, include_1h=True, include_2h=False, **kwargs)
C2h = pspec.ang_power_spectrum(cosmo, l, profiles, include_1h=False, include_2h=True, **kwargs)

plt.figure()
plt.loglog(l, Cl, "b")
plt.loglog(l, C1h, "r")
plt.loglog(l, C2h, "g")